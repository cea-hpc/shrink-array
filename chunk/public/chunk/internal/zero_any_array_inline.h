/*
 * Copyright 2024-2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <atomic>        // std::memory_order (the missed-compression audit)
#include <chrono>        // std::chrono::steady_clock (rechunk=debug: the registry release time)
#include <cstddef>       // std::size_t
#include <array>         // std::array (the residual's sample)
#include <format>        // std::format (print_type_name)
#include <iostream>      // std::cout (print_type_name)
#include <string_view>   // std::string_view (the residual's kind)
#include <string>        // the type name in print_type_name
#include <type_traits>   // std::remove_reference_t
#include <utility>       // std::pair (the clear() summary)
#include <vector>        // std::vector
#include <algorithm>     // std::min
#include <cstdint>       // std::uint8_t
#include <ratio>         // std::milli
#include <unordered_set> // std::unordered_set

#include "chunk/zero_any_array.h"                       // zero_any::ArrayImpl, the class these bodies belong to
#include "chunk/internal/type_state_registry.h"         // TypeStateRegistry::ensure_manager
#include "chunk/internal/chunk_size_predictor.h"        // report_size_prediction
#include "chunk/log.h"                                  // CHUNK_LOG, logging::enabled (the size prediction's gate)
#include "chunk/internal/chunk_index_sample.h"          // chunk_index_sample (the residual's sample)
#include "chunk/internal/chunk_stats.h"                 // inc_chunk_* (CHUNK_STATS)
#include "chunk/internal/chunk_stats_printer.h"         // print_formatted_chunk_stats
#include "chunk/internal/type_state.h"                  // TypeState: per-size free lists, candidate filter
#include "chunk/internal/pmp_chunk.h"                   // CHUNK_PMP_*, pmp loops
#include "chunk/internal/singleton.h"                   // Singleton<TypeStateRegistry>::get_ref
#include "chunk/internal/type_name.h"                   // type_name<T>
#include "chunk/internal/invariant_check.h"             // CHUNK_INVARIANT_FAILED
#include "chunk/internal/read_reclaim.h"                // read_reclaim::PassScope (drain_own_candidates)
#include "chunk/internal/array_users.h"                 // ArrayUsers::only (drain_own_candidates)
#include "chunk/internal/own_candidates.h"              // with_own_candidates_only (drain_own_candidates)
#include "chunk/config.h"                               // build_config::chunk_dynamic, chunk_config::stats
#include "chunk/internal/chunk_size_policy.h"           // set_thread_chunk_size
#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement
#include "chunk/internal/parallel_storage_init.h"       // for_each_chunk
#include "chunk/internal/pool_ledger.h"                 // CHUNK_LEDGER_* (check builds)
#include "chunk/internal/cold_path_lock.h"              // ColdPathLock: construction, teardown, resize
#include "chunk/internal/teardown_probe.h"              // teardown_probe (rechunk=debug: the tmp teardown split)
#include "chunk/internal/teardown_park.h"               // park_live_chunks (the park step)
#include "chunk/internal/same_value.h"                  // same_value

namespace shrink_array
{

template <typename Derived>
void zero_any::ArrayImpl<Derived>::rebuild_storage(IntType sz)
{
    if (sz)
    {
        if (this->size_policy_.size())
        {
            chunk_count_ = this->chunks_for(sz);
        }
        else if (sz && !this->size_policy_.size())
        {
            CHUNK_INVARIANT_FAILED("create({} elements): the chunk size is 0 (no geometry set)", sz);
        }

        if (chunk_count_)
        {
            Manager::inc_total_chunk_create(chunk_count_);
            Manager::compute_size_buffer_reuse();
            Manager::compute_size_max_number_of_chunk_to_check_compress();

            acquire_second_value_registry();
            this->acquire_chunk_list(static_cast<std::size_t>(chunk_count_));

            this->alloc_dirty_bitset(chunk_count_);

            // old=0: full (re)build of the metadata tables. Correct because every
            // entry is reinitialised just below, BEFORE any refill, so no proxy
            // holds a cached ptr_second_value into a table being reallocated.
            ProxyType::resize_array_is_int(array_id_, 0, chunk_count_);
            this->assign_alloc_key(); // own slabs if large enough (see alloc_key_)

            // Per-chunk init: every iteration writes slot i of tables sized just
            // above and nothing else, so it parallelises as is (a serial loop is
            // a visible master-only cost on large arrays). Same pattern as
            // zero_or_value::Array::rebuild_storage; serial inside a region (the caller
            // need not be the whole team) and below the one threshold,
            // parallel_storage_init::parallel_init_min_chunks.
            const auto init_chunk = [&](int chunk_index)
            {
                ProxyType::Base::set_candidate_flag(array_id_, chunk_index, 0);
                ProxyType::array_type_const_val[array_id_][chunk_index] = ConstSlot::first;
                ProxyType::BaseSpec::set(array_id_, chunk_index, T{});
                this->construct_chunk(chunk_index).init_after_constructor();
            };
            parallel_storage_init::for_each_chunk(chunk_count_, init_chunk);
        }
    }

    m_size_ = sz;
}

/// @brief Print the demangled name of T and @p n_second_values on std::cout.
/// @param n_second_values Number of compression values to report.
template <ArrayChunkElement T>
void print_type_name(int n_second_values)
{
    // std::format, not std::left/std::setw: std::cout's flags stay as they were.
    std::cout << std::format("{:<30} → Number of compression values : {}\n", "Type: " + type_name<T>(),
                             n_second_values);
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::print_compression_stats(std::ostream& os) const
{
    if (chunk_count_ == 0)
    {
        return;
    }

    // Do not classify from array_type_const_val alone: it is written only at
    // construction and never reset on materialisation, so a MATERIALISED chunk
    // would be counted as compressed. written() tests the LIVE state
    // (values_load() != first_value && is_not_second_value()); array_type_const_val is
    // trusted only for chunks that are NOT written.
    // A written chunk is then re-tested on its CONTENT (constant_value(), the
    // same scan as the size predictor): written() only says whether the
    // opportunistic deferred compression happened to catch the chunk, which is
    // thread-count dependent. Classifying on content makes still@exit a
    // function of the data alone, matching the sequential build's eager
    // compression.
    //
    // Each chunk is classified on its own, so the scan is shared out over the
    // team (the counts are reductions). Same guard as the other teardown loops:
    // never nest inside a caller's region, no team for a handful of chunks.
    const auto csz_audit = static_cast<unsigned int>(this->size_policy_.size());

    // acc[0]: compressed (or compressible) to first_value, acc[1]: to a second value.
    const auto counts = pmp::parallel_sum<int, 2>(
        0, chunk_count_, pmp::Options{.use_team = !CHUNK_PMP_IN_PARALLEL_REGION && chunk_count_ > 500},
        [&](int j, auto& acc)
        {
            if (const auto& ck = chunk_list_[j]; ck.written())
            {
                const auto v0 = ck.constant_value(csz_audit);
                if (!v0)
                {
                    return;
                }
                ++acc[same_value(*v0, static_cast<T>(ChunkType::first_value)) ? 0 : 1];
                return;
            }
            const ConstSlot ct = ProxyType::array_type_const_val[array_id_][j];
            ++acc[ct == ConstSlot::first ? 0 : 1];
        });
    const int n_t1         = counts[1];
    const int n_compressed = counts[0] + n_t1;
    if constexpr (chunk_config::stats)
    {
        add_chunk_still_compressed_at_exit<ChunkType>(n_compressed);
        record_chunk_size_at_exit<ChunkType>(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Shared layout/colours/verdict: see chunk_stats_printer.h.
    print_formatted_chunk_stats(os, array_id_, this->debug_name(), type_name<T>(), n_compressed, chunk_count_, n_t1,
                                "zero_any",
                                /*n_cross=*/-1, /*cross_name=*/"", /*bool_elements=*/std::is_same_v<T, bool>);
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::clear()
{
    this->invalidate_proxies();
    // Chunk-size predictor (log only, no re-chunk; any_check only): scans the
    // chunks once, logs the current vs predicted-optimal size. Runs before the
    // drain, while the chunk state is still valid. One summary slot per chunk,
    // written by index, so the scan is shared over the team (same guard as the
    // stats scan) and keeps chunk order.
    if constexpr (build_config::any_check)
    {
        // The scan only feeds the advisor's info line: skipped when it is not shown.
        // Inside try: clear() runs from the destructor (destroy_storage()), which
        // is noexcept, and the summary's allocation or the report's formatting can
        // throw (std::bad_alloc). A log line: given up, never std::terminate()
        // (Coverity 661905, 661906).
        if (chunk_count_ > 0 && logging::enabled(logging::Channel::advisor, logging::Level::info))
        {
            try
            {
                const auto                      csz = static_cast<unsigned int>(this->size_policy_.size());
                std::vector<std::pair<bool, T>> summary(static_cast<std::size_t>(chunk_count_));
                pmp::parallel_for(0, chunk_count_,
                                  pmp::Options{.use_team = !CHUNK_PMP_IN_PARALLEL_REGION && chunk_count_ > 500},
                                  [&](int j)
                                  {
                                      const auto& ck = chunk_list_[j];
                                      if (!ck.written())
                                      {
                                          if constexpr (has_motif_state_v<ChunkType>)
                                          {
                                              // A motif chunk is no constant (stage 1: no advice on motifs).
                                              if (ChunkType::is_motif_ptr(ck.values_load()))
                                              {
                                                  summary[static_cast<std::size_t>(j)] = {false, T{}};
                                                  return;
                                              }
                                          }
                                          summary[static_cast<std::size_t>(j)] = {true, ck.read_value_scan(0)};
                                          return;
                                      }
                                      const auto v0                        = ck.constant_value(csz);
                                      summary[static_cast<std::size_t>(j)] = {v0.has_value(), v0.value_or(T{})};
                                  });
                const auto& registry    = *ProxyType::array_registry;
                const int   array_index = registry.get_array_id(this->registry_key());
                report_size_prediction<T>(summary, csz, array_index, "zero_any",
                                          sizeof(std::remove_reference_t<decltype(chunk_list_[0])>));
            }
            catch (...) // NOSONAR(cpp:S2486, cpp:S2738): teardown, nothing may escape
            {
                // The prediction is a log line only: skipped.
            }
        }
    }

    // Everything below (drain, terminal compression, buffer release, metadata
    // reset) is the storage teardown shared with rechunk_to().
    clear_core();
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::destroy_storage()
{
    const internal::ColdPathLock cold_path_lock(internal::cold_path_mutex());
    clear();
    ProxyType::array_registry->erase(this->registry_key());
    // rechunk=debug, tmp teardown only (armed by rechunk_to): free chunk_list_'s
    // buffer here rather than with the member, so that the split can time it.
    // The array is being destroyed: the same release, a few instructions earlier.
    auto& probe = internal::teardown_probe();
    if (probe.on)
    {
        probe.mark(internal::TeardownStep::erase);
        typename Base::ChunkListType().swap(chunk_list_);
        probe.mark(internal::TeardownStep::list_unmap);
    }
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::clear_core()
{
    const internal::ColdPathLock cold_path_lock(internal::cold_path_mutex());
    // Pin the ambient size to THIS array's own geometry before the sweep
    // below. Same rule as zero_or_value::Array::clear_core() -- see that
    // function's comment.
    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }
    // rechunk=debug, tmp teardown only (armed by rechunk_to): the split's marks.
    auto& probe = internal::teardown_probe();
    probe.mark(internal::TeardownStep::entry);

    // Drain any deferred write still targeting this array's chunks before
    // chunk_list_ is freed below. force=true (see
    // ParallelSupportCommon::flush_pending_deferred_ops): the non-force path is
    // gated on CM::any_pending_ops() and on the per-chunk dirty bit, and when
    // either misreads NOTHING is flushed -- the audit below then scans a chunk
    // whose writes are still queued, constant_value() (raw values_load()) sees a
    // non-constant buffer, and the chunk is under-counted as still-compressed.
    // zero_any carries far more deferred traffic than zero/zero_or_value, so it exposes this first.
    // Inside a region, a work array of this thread's: this rank's structures only.
    const bool own_rank = this->teardown_on_own_rank(array_id_);
    this->flush_pending_deferred_ops(/*force=*/true, own_rank);
    probe.mark(internal::TeardownStep::flush);

    // Per-array compression stats: read after the flush (before it, in-flight
    // async compression moves the written() state, a thread-count-dependent
    // snapshot) and before the terminal compression/release sweep below mutates it.
    if (this->exit_census_)
    {
        print_compression_stats();
    }

    // Remove this array's chunks from the shared compression candidate list
    // before they are freed: in non-OMPT builds nothing else clears it, and a
    // later compression pass would walk the dangling Chunk* (use-after-free).
    // By array_id, so other live arrays' candidates are untouched.
    Manager::remove_from_zero_any_candidates_by_array_id(array_id_, own_rank);

    // Also drop any candidate whose chunk pointer lies in THIS array's
    // chunk_list_ span: the by-array_id removal misses a tuple whose array_id was
    // shifted by a rechunk swap, and a later compress pass would double-free its
    // buffer. Must run before the terminal sweep and before chunk_list_.clear().
    if (!chunk_list_.empty())
    {
        Manager::remove_from_zero_any_candidates_by_ptr_range(
            static_cast<const void*>(chunk_list_.data()),
            static_cast<const void*>(chunk_list_.data() + chunk_list_.size()), own_rank);
    }
    probe.mark(internal::TeardownStep::candidates);

    if (!chunk_count_)
    {
        m_size_ = 0;

        chunk_list_.clear();
        ProxyType::clear_array_is_int(array_id_);

        return;
    }

    Manager::dec_total_chunk_create(chunk_count_);
    // Recompute what derives from the total -- see zero::Array::clear_core():
    // otherwise a re-chunk leaves the threshold sized for the geometry it just left.
    Manager::compute_size_buffer_reuse();
    Manager::compute_size_max_number_of_chunk_to_check_compress();

    if constexpr (chunk_config::not_use_compression_with_ompt)
    {
        // Skip this terminal sweep in aggressive mode (compression is left to the
        // in-loop perform_compression -> make_block_compression path): there,
        // compress()'s clear_int_st calls are compiled out, so the sweep would
        // only scan every chunk's values without compressing anything -- a cost
        // that dominates the profile.
        if (!chunk_config::aggressive_compression()) // run-time switch: chunk/config.h
        {
            const auto& type_registry = Singleton<TypeStateRegistry>::get_ref();

            // Not for a work array torn down in a region: that pass walks every
            // rank's list, of every type.
            if (!own_rank && type_registry.has_candidates())
            {
                type_registry.run_compress_pass(false);
            }
        }

        // ── Missed-compression audit ─────────────────────────────────────────────
        // Check + non-aggressive only. After the terminal pass, walk THIS array's
        // chunks (not the candidate list, so never-enqueued blocks are seen) for
        // every block (a) still materialised (written()), (b) never registered as
        // a candidate (candidate_flags == 0), yet (c)
        // constant. Two different things, told apart by the value:
        //
        //   MISSED  constant at first_value, or at the chunk's own second value
        //           (get_array_second_value): the pass would have compressed it,
        //           the write path let it become constant without queueing it.
        //           A few is benign (a straggler), a recurring/large count means
        //           the marking path has a hole.
        //   STUCK   constant at ANOTHER value: zero_any keeps a chunk's second
        //           value until the chunk passes through first_value (the clear
        //           to the first value resets it), so a chunk compressed to V once and
        //           uniformly W later is not compressible as it stands -- not by
        //           the write path, not by compress(). This count is the
        //           memory that design choice costs on this array -- not a
        //           marking miss.
        //
        // Non-fatal; O(chunk_count * chunk_size) once at teardown, check builds only.
        if constexpr (build_config::any_check)
        {
            if (!chunk_config::aggressive_compression()) // run-time switch: chunk/config.h
            {
                const auto  csz           = static_cast<unsigned int>(this->size_policy_.size());
                const auto& inside        = ProxyType::candidate_flag_row(array_id_);
                std::size_t missed_first  = 0;
                std::size_t missed_second = 0;
                std::size_t stuck         = 0;

                for (int j = 0; j < chunk_count_; ++j)
                {
                    const auto& ck = chunk_list_[j];
                    if (!ck.written())
                    {
                        continue; // already constant/compressed — nothing owed
                    }
                    if (inside[j].load(std::memory_order_acquire) != 0)
                    {
                        continue; // legitimately queued (a straggler the pass left) — not "missed"
                    }

                    // Is it actually constant? (same notion classify_candidate uses:
                    // every element equal to the block's first element.)
                    const auto constant = ck.constant_value(csz);
                    if (!constant)
                    {
                        continue;
                    }
                    if (same_value(*constant, static_cast<T>(ChunkType::first_value)))
                    {
                        ++missed_first;
                    }
                    else if (same_value(
                                 *constant,
                                 static_cast<T>(ProxyType::BaseSpec::get_array_second_value(array_id_, j).get())))
                    {
                        ++missed_second;
                    }
                    else
                    {
                        ++stuck;
                    }
                }

                if (missed_first + missed_second != 0)
                {
                    CHUNK_LOG(info, compress,
                              "audit: zero_any array={}: {} constant block(s) never queued for compression ({} at "
                              "first_value, {} at their second value: missed by the marking path). A few is benign; a "
                              "recurring/large count is not.",
                              array_id_, missed_first + missed_second, missed_first, missed_second);
                }
                if (stuck != 0)
                {
                    CHUNK_LOG(
                        info, compress,
                        "audit: zero_any array={}: {} block(s) constant at a value other than their second value: "
                        "stuck, not a marking miss (zero_any keeps a chunk's second value until it passes "
                        "through first_value).",
                        array_id_, stuck);
                }
            }
        }
    }

    probe.mark(internal::TeardownStep::terminal);

    if constexpr (build_config::not_threaded)
    {
        for (int j = 0; j < chunk_count_; ++j)
        {
            if (chunk_list_[j].written())
            {
                if constexpr (chunk_config::stats)
                {
                    inc_chunk_teardown_clear<ChunkType>();
                }
                chunk_list_[j].clear_st(&ProxyType::array_type_const_val[array_id_][j],
                                        ProxyType::BaseSpec::get_array_second_value(array_id_, j));
            }
        }

        // Give back the registry reference this array took
        // (acquire_second_value_registry), as the OpenMP branch does after its barrier;
        // without it the last array out never frees the registry.
        if (holds_second_value_registry_)
        {
            int n_second_values = 0;
            ChunkType::Holder::clear(n_second_values); // single thread here
            holds_second_value_registry_ = false;
            if (n_second_values != 0)
            {
                print_type_name<T>(n_second_values);
            }
        }
    }
    else
    {
        // rechunk=debug: the registry release's own time, measured by whichever
        // thread runs the single below and read after the region's join.
        const bool time_registry = probe.on;
        double     registry_ms   = 0.0;
        // acc[0]: the second values the registry release counted (one thread's).
        const int n_second_values = pmp::team_sum<int, 1>(
            pmp::Options{},
            [&](pmp::Team& team, auto& acc)
            {
                // Neutralised to first_value: every compress path short-circuits
                // (the CAS reads values == first_value and returns nullptr). Mirrors
                // the non-OMP clear_st path.
                internal::park_live_chunks<T>(
                    team, chunk_list_, chunk_count_, this->size_policy_.size(),
                    [](const ChunkType& chunk) { return chunk.written_test(); },
                    [](ChunkType& chunk) { chunk.values_store(ChunkBase<T, ChunkType>::first_value_ptr()); });

                // BARRIER REQUIRED: park_live_chunks() ends on a for_static(..., nowait), so other threads may
                // still be in it writing values_, whose slow path (make_block_compression ->
                // find_or_register_second_value) SCANS the shared segment list. clear() below tears
                // that list down (head.exchange(nullptr) + delete of every Segment); without
                // the barrier the teardown races the scan (freed-Segment dereference).
                team.barrier();
                team.single(
                    [&]
                    {
                        // Only the reference this array took (acquire_second_value_registry).
                        if (holds_second_value_registry_)
                        {
                            const auto r0 = time_registry ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};
                            ChunkType::Holder::clear(acc[0]);
                            holds_second_value_registry_ = false;
                            if (time_registry)
                            {
                                registry_ms =
                                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0)
                                        .count();
                            }
                        }
                    });
            })[0];
        probe.mark(internal::TeardownStep::park);
        probe.transfer(internal::TeardownStep::park, internal::TeardownStep::registry, registry_ms);

        if (n_second_values != 0)
        {
            print_type_name<T>(n_second_values);
        }
    }

    m_size_ = 0;

    this->release_chunk_list(); // required by the allocator

    ProxyType::clear_array_is_int(array_id_);

    chunk_count_ = 0;
    probe.mark(internal::TeardownStep::tables);

    this->drain_own_reuse_list(); // an array with its own allocator set: its memory goes back now
    probe.mark(internal::TeardownStep::key_drain);
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::compress_int(pmp::Team& team)
{
    // Pin the ambient size to THIS array's geometry: the frees below route on
    // thread_chunk_size(), not on the buffer (see clear_core()); left to the
    // array the thread last served, a re-chunked array's buffers would go to
    // another size's allocator.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    std::vector<int> chunks_at_first;
    std::vector<int> chunks_at_second;

    const int n_res = (chunk_count_ / (50 * CHUNK_PMP_MAX_THREADS)) + 1;
    chunks_at_first.reserve(n_res);
    chunks_at_second.reserve(n_res);

    team.for_static<int>(
        0, chunk_count_,
        [&](int j)
        {
            const auto& chunk = chunk_list_[j];

            // Through SecondValueRef::get(), not the raw slot: under
            // CHUNK_SECOND_VALUE_INDEXED a slot holds the value's IDENTIFIER, and
            // `const T v = slot` would silently convert that integer into a T.
            const T second_value = ProxyType::BaseSpec::get_array_second_value(array_id_, j).get();

            if (chunk.written())
            {
                // second_value == first_value comes back as `first` only.
                const auto scan = chunk.scan_constants(ChunkType::first_value, second_value,
                                                       static_cast<unsigned int>(this->size_policy_.size()));
                if (scan.first)
                {
                    chunks_at_first.push_back(j);
                }
                if (scan.second)
                {
                    chunks_at_second.push_back(j);
                }
            }
        },
        pmp::nowait);

    // ── What this reports, and what it does NOT assert ───────────────────────
    // compress_int() only SCANS; it never compresses (unlike zero::Array's, whose
    // non-embedded branch does the clear itself). It reports the residual:
    // blocks constant right now and still materialised.
    //
    // Do not treat that residual as an invariant violation: compression is
    // incremental (a write only QUEUES its chunk; make_block_compression / the
    // phase-end pass drains the queue), so a block that just became constant
    // normally waits for the next pass.
    //
    // What matters is its SIZE and TREND: a handful is the normal in-flight
    // residual; a large or growing count means the incremental path is not
    // keeping up (or, in a threaded && !OMPT build, that Chunk::compress()
    // returns early by design and make_block_compression is the only
    // compressor left -- see its guard in zero_any_chunk_inline.h).
    // compress=info, one line per kind and thread (its share of the chunks).
    const auto report = [&](const std::vector<int>& chunks, std::string_view at)
    {
        if (chunks.empty())
        {
            return;
        }
        std::array<char, 256> sample{};
        CHUNK_LOG(info, compress,
                  "residual Array<{}, {}>: {} block(s) constant at {}, still materialised (awaiting the next "
                  "compression pass): {}",
                  type_name<T>(), type_name<IntType>(), chunks.size(), at,
                  internal::chunk_index_sample(sample, chunks, static_cast<IntType>(this->size_policy_.size())));
    };
    report(chunks_at_first, "first_value");
    report(chunks_at_second, "the second value");
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::make_block_compression()
{
    if (!block_pass_refused_in_region())
    {
        drain_candidates();
    }
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::drain_candidates()
{
    // No region opened here, no writer handshake (nullptr proxy): a sequential
    // point, every rank's list, drained serially by the caller. Inside a
    // region, resize_common() calls drain_own_candidates() instead.
    for (int thr = 0; thr < CHUNK_PMP_RANK_BOUND; ++thr)
    {
        ChunkType::make_block_compression(thr, nullptr);
    }
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::drain_own_candidates()
{
    // An array resized inside a region by its owning thread. No writer
    // handshake either: nobody else writes THIS array. The thread's list also
    // holds chunks of arrays other threads are writing: they sit the pass out.
    // The buffers released wait for one reader grace period (read_reclaim.h).
    const int own = CHUNK_PMP_THREAD_RANK;
    // A work array of the calling thread: its own list. A SHARED array (other
    // ranks queued chunks of it: ArrayUsers) is resized at a sequential point
    // -- under `omp single`, the team waiting -- and every rank's list holds
    // &chunk_list_[j] pointers and set flags that the resize invalidates:
    // every rank's list then, this array's candidates only as well.
    const bool shared = !internal::ArrayUsers<ChunkType>::only(array_id_, own);
    const int  first  = shared ? 0 : own;
    const int  last   = shared ? CHUNK_PMP_RANK_BOUND : own + 1;
    for (int rank = first; rank < last; ++rank)
    {
        internal::with_own_candidates_only(Manager::get_zero_any_candidates(rank), array_id_,
                                           [rank]
                                           {
                                               const read_reclaim::PassScope pass_scope;
                                               ChunkType::make_block_compression(rank, nullptr);
                                           });
    }
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::release_chunk_range(int first, int last)
{
    if (first >= last)
    {
        return;
    }
    // The release routes by the ambient geometry: pin it to this array's.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    if constexpr (build_config::not_threaded)
    {
        // Before Proxy::resize_array_is_int() truncates the per-chunk type
        // table: clear_st() reads array_type_const_val[array_id][j].
        const int array_id = ProxyType::array_registry->get_array_id(this->registry_key());
        for (int j = first; j < last; ++j)
        {
            if (chunk_list_[j].written())
            {
                if constexpr (chunk_config::stats)
                {
                    inc_chunk_teardown_clear<ChunkType>();
                }
                chunk_list_[j].clear_st(&ProxyType::array_type_const_val[array_id][j],
                                        ProxyType::BaseSpec::get_array_second_value(array_id, j));
            }
        }
    }
    else
    {
        // A large shrink takes a team of its own, as clear_core()'s release
        // does: each thread parks into ITS rank's reuse list of this array's
        // geometry. Inside a region, or for a few chunks, the calling thread.
        const unsigned int my_sz = this->size_policy_.size();
        parallel_storage_init::for_each_chunk_in(
            first, last,
            [my_sz]() -> auto& { return Manager::get_list_block_free_thr_for_size(CHUNK_PMP_THREAD_RANK, my_sz); },
            [this](int j, auto& lbf)
            {
                if (chunk_list_[j].written_test())
                {
                    if constexpr (chunk_config::stats)
                    {
                        inc_chunk_teardown_clear<ChunkType>();
                        // The size read here, not captured: stats builds only use it.
                        add_chunk_released<ChunkType>(1, static_cast<long>(sizeof(T)) *
                                                             static_cast<long>(this->size_policy_.size()));
                    }
                    T* values_loc = chunk_list_[j].values_;
                    CHUNK_LEDGER_PARK(values_loc, "resize.shrink_park");
                    lbf.push_back(values_loc);
                    // Neutralise, as clear_core(): the buffer belongs to the
                    // reuse list now.
                    chunk_list_[j].values_store(ChunkBase<T, ChunkType>::first_value_ptr());
                }
            });
    }
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::resize(IntType newsz)
{
    this->check_length(newsz, "resize");
    const internal::ColdPathLock cold_path_lock(internal::cold_path_mutex());
    this->invalidate_proxies();
    const auto& registry = *ProxyType::array_registry;

    const int     array_id        = registry.get_array_id(this->registry_key());
    const int     old_chunk_count = chunk_count_;
    const IntType old_size        = this->m_size_; // the stale-tail reset at the end

    SecondValueBase::resize_common(newsz);
    if (chunk_count_ > 0)
    {
        acquire_second_value_registry(); // an array built empty gets its chunks here
    }

    ProxyType::resize_array_is_int(array_id, old_chunk_count, chunk_count_);

    // The new chunks [old_chunk_count, chunk_count) and their table slots: a team of
    // its own outside a region, the calling thread inside one. Never an
    // orphaned `omp for`: inside a region it would construct only 1/N of a
    // thread's own array. Each iteration writes slot i of tables sized above.
    parallel_storage_init::for_each_chunk(chunk_count_ - old_chunk_count,
                                          [this, old_chunk_count, array_id](int k)
                                          {
                                              const int chunk_index = old_chunk_count + k;
                                              this->construct_chunk(chunk_index).init_after_constructor();
                                              ProxyType::Base::set_candidate_flag(array_id, chunk_index, 0);
                                              ProxyType::array_type_const_val[array_id][chunk_index] = ConstSlot::first;
                                              // T{}: the struct elements have no conversion from 0
                                              ProxyType::BaseSpec::set(array_id, chunk_index, T{});
                                          });

    // The regrown range [old_size, newsz) reads T{}, as std::vector's.
    this->reset_regrown_tail(old_size, newsz, old_chunk_count);
}

// ---------------------------------------------------------------------------
// shrink_array::ArrayImpl<Derived>
// ---------------------------------------------------------------------------

template <typename Derived>
bool zero_any::ArrayImpl<Derived>::operator==(const ArrayImpl& other) const noexcept
{
    if (m_size_ != other.m_size_)
    {
        return false;
    }
    if (m_size_ == 0)
    {
        return true;
    }
    const auto  csz = static_cast<long long>(this->chunk_size());
    std::atomic differ{false};
    parallel_storage_init::for_each_chunk(this->chunk_index_of(m_size_ - 1) + 1,
                                          [&](int c)
                                          {
                                              if (differ.load(std::memory_order_relaxed))
                                              {
                                                  return;
                                              }
                                              const long long lo = static_cast<long long>(c) * csz;
                                              const long long hi = std::min(lo + csz, static_cast<long long>(m_size_));
                                              for (long long i = lo; i < hi; ++i)
                                              {
                                                  if (SecondValueBase::read_value(static_cast<IntType>(i)) !=
                                                      other.read_value(static_cast<IntType>(i)))
                                                  {
                                                      differ.store(true, std::memory_order_relaxed);
                                                      return;
                                                  }
                                              }
                                          });
    return !differ.load(std::memory_order_relaxed);
}

template <typename Derived>
std::size_t zero_any::ArrayImpl<Derived>::reclaim_second_values()
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return 0; // one shared constant array, no registry (SecondValueRegistry<bool>)
    }
    else
    {
        using Gvh = GlobalVecHolder<T>;
        // Each refusal counted by cause (second_value_stats()): it returns
        // 0 as a call with nothing to free does.
        if (CHUNK_PMP_IN_PARALLEL_REGION)
        {
            Gvh::refused_parallel.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        if (g_in_rechunk_rebuild.load(std::memory_order_acquire))
        {
            Gvh::refused_rechunk.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        if (Gvh::index_degraded.load(std::memory_order_relaxed))
        {
            Gvh::refused_degraded.fetch_add(1, std::memory_order_relaxed);
            // Sticky until the last array of the type goes: say it once, the
            // run will otherwise end on the identifier exhaustion with no cause.
            static std::atomic warned{false};
            if (!warned.exchange(true, std::memory_order_relaxed))
            {
                CHUNK_LOG(warn, compress,
                          "reclaim_second_values() refused: the second-value index is degraded, a value unfindable "
                          "by value could be freed. Nothing is reclaimed for this type until its last array is "
                          "destroyed.");
            }
            return 0;
        }
        // Check builds: no lookup or registration in flight from here to the
        // end of reclaim_unmarked() -- the mark included, since a slot found
        // before it and recorded after it would be freed (GlobalVecHolder's
        // "Reclaim contract").
        const typename Gvh::ReclaimScope scope;
        Gvh::reclaim_runs.fetch_add(1, std::memory_order_relaxed);
        const std::size_t handed_out = Gvh::ids_handed_out();
        if (handed_out == 0)
        {
            return 0;
        }
        std::vector<std::uint8_t> referenced(handed_out + 1, 0);

        // What every member of the family sharing the registry still names.
        Gvh::mark_all(referenced);

        const std::size_t freed = Gvh::reclaim_unmarked(referenced);
        Gvh::reclaim_freed.fetch_add(freed, std::memory_order_relaxed);
        return freed;
    }
}

template <typename Derived>
[[nodiscard]] SecondValueStats zero_any::ArrayImpl<Derived>::second_value_stats() noexcept
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return {};
    }
    else
    {
        return GlobalVecHolder<T>::stats();
    }
}

template <typename Derived>
inline void zero_any::ArrayImpl<Derived>::acquire_second_value_registry()
{
    if (!holds_second_value_registry_)
    {
        if constexpr (!std::is_same_v<T, bool>)
        {
            GlobalVecHolder<T>::register_marker(&mark_second_values<T, ProxyType, ChunkType>);
        }
        ChunkType::Holder::create();
        holds_second_value_registry_ = true;
    }
}

template <typename Derived>
void zero_any::ArrayImpl<Derived>::gc_collect_live(std::unordered_set<const void*>& live)
{
    ProxyType::array_registry->for_each_key(
        [&](ArrayKey<ChunkType> arr)
        {
            // The key is the array's core, whatever its index type.
            const int nc = arr->chunk_count();
            for (int j = 0; j < nc; ++j)
            {
                const auto& ch = arr->chunk_ref(static_cast<std::size_t>(j));
                if (ch.is_second_value())
                {
                    live.insert(static_cast<const void*>(ch.values_load()));
                }
            }
        });
}

// ---------------------------------------------------------------------------
// shrink_array::Array<T, IntType>
// ---------------------------------------------------------------------------

template <ArrayChunkElement T, typename IntType>
zero_any::Array<T, IntType>& zero_any::Array<T, IntType>::operator=(Array&& other) noexcept
{
    if (this != &other)
    {
        this->clear();
        this->swap_storage_and_identity(other);
    }
    return *this;
}

} // namespace shrink_array
