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
/**
 * @file proxy_base.h
 * @brief Common base class for Proxy2 and Proxy2A using CRTP pattern
 *
 * ProxyBase holds all common functionality; derived classes keep only
 * chunk-specific logic.
 */

#include <type_traits> // std::is_integral_v, std::is_same_v, std::conditional_t, std::remove_reference_t
#include <atomic>      // std::atomic
#include <complex>     // std::complex
#include <cstdlib>     // std::size_t
#include <functional>  // std::plus, std::minus
#include <vector>      // std::vector

#include "chunk/internal/element_access.h" // elem_get / elem_set, is_packed_bit_v: Bit buffers are words of bits
#include "chunk/config.h"                  // chunk_config::use_fast_block_compression, build_config::any_check
#include "chunk/internal/chunk_stats.h" // increm_chunk_* in the bodies (dependent calls the include tools cannot attribute)
#include "chunk/internal/def_force_inline.h"             // CHUNK_FORCE_INLINE: the write engine and its wrappers
#include "chunk/internal/in_t.h"                         // in_t<T>
#include "chunk/internal/invariant_check.h"              // CHUNK_INVARIANT_FAILED
#include "chunk/internal/proxy_operators.h"              // ProxyOperators
#include "chunk/internal/tab_cst_val2_holder.h"          // TabCstVal2Holder: second-value identifiers
#include "chunk/internal/relaxed_element.h"              // store_value: Chunk2A's relaxed element stores
#include "chunk/internal/concept_is_clean_assignement.h" // IsCleanCompoundOp, IsCleanShiftOp, base_t
#include "chunk/internal/crp_helper.h"                   // CRTPSupport, CHUNK_CRTP_USE_DEDUCING_THIS
#include "chunk/internal/test_hooks.h"                   // TEST KNOB: CHUNK_TEST_READ_DELAY_EVERY (read_values)
#include "chunk/internal/read_reclaim.h"                 // ReadSection: read_values_outside_session
#include "chunk/internal/immortal.h"                     // Immortal / LazyImmortal: statics safe for global arrays
#include "chunk/internal/parallel_storage_init.h"        // no_init_vector, make_filled: the candidate-flag rows

namespace shrink_array
{
/// @brief One array's per-chunk candidate flags ("already in a compression
///        list"): one atomic<char> per chunk, built by make_filled (a team for
///        a large array). char, not bool: vector<bool> is the packed
///        specialisation.
using ChunkFlagRowT = parallel_storage_init::NoInitVector<std::atomic<char>>;

// Forward declaration: only named here, in contexts that depend on T. The
// definition (chunk/internal/compression_manager.h) comes with the _inline tails.
template <typename T, typename ChunkType>
struct CompressionManager;

template <typename T>
struct ArithmeticProxyContext;

template <typename T, typename Op>
struct ArithmeticOperationProxyContext;

template <typename T>
class Proxy;

/// @brief Whether T is a Proxy\<U\> (Proxy2 and Proxy2A are not): false by default.
template <typename T>
struct is_simple_proxy : public std::false_type
{
};

/// @brief Proxy<T> is a simple proxy.
template <typename T>
struct is_simple_proxy<Proxy<T>> : public std::true_type
{
};
namespace internal
{
/// @brief Element @p n behind a values pointer the caller has loaded -- the
///        raw read of ProxyBase::read_values() and read_values_async().
/// @details NOT raw for a Chunk2A second constant: its pointer names an
///          identifier in a region that is never written, so indexing it reads
///          a zero page; the value comes from the registry instead (arithmetic
///          plus one table load, no sync). Ask the POINTER, not the chunk: the
///          callers run with a write or a replay in flight, so the chunk's state
///          and the pointer can disagree, and converting a pointer outside the
///          region indexes value_by_id[] out of bounds.
template <typename T>
[[nodiscard]] T element_of(const T* vals, std::size_t n) noexcept
{
    if constexpr (requires(const T* ptr) {
                      TabCstVal2Holder<T>::is_cst_val2_ptr(ptr);
                      TabCstVal2Holder<T>::value_of_ptr_checked(ptr);
                  })
    {
        if (TabCstVal2Holder<T>::is_cst_val2_ptr(vals)) [[unlikely]]
        {
            return TabCstVal2Holder<T>::value_of_ptr_checked(vals);
        }
    }
    // elem_get, not vals[n]: for Bit, vals[n] indexes BYTES past the
    // chunk_size / 8 byte buffer.
    // raw-read: element_of -- callers hold the write session, a ReadSection, or replay after materialisation
    return elem_get(vals, n);
}

/// @brief Check builds: stop on an integer `/=` or `%=` by zero.
/// @details Integer division or modulo by zero is undefined behaviour in C++
///          (SIGFPE on x86, a silent 0 on ARM); an invariant failure says
///          where. Integer elements only: for a floating T the division is
///          IEEE (inf, NaN) and the result is defined. Compiled out of release
///          builds. Shared by Proxy (v1) and Proxy2Base (Chunk2, Chunk2A).
/// @tparam T Element type.
/// @param v Divisor (a value or a proxy).
/// @param op "/=" or "%=", for the message.
template <typename T, typename U>
CHUNK_FORCE_INLINE void check_integer_divisor([[maybe_unused]] const U& v, [[maybe_unused]] const char* op) noexcept
{
    if constexpr (build_config::any_check && std::is_integral_v<T> && !std::is_same_v<T, bool>)
    {
        if (static_cast<BaseT<U>>(v) == BaseT<U>{}) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("integer element {} by zero (undefined behaviour)", op);
        }
    }
}
} // namespace internal

/**
 * @brief The compression-bookkeeping half of ProxyBase.
 *
 * @tparam Derived The concrete proxy (CRTP).
 * @tparam T Element type.
 * @tparam ChunkType The chunk this proxy points into.
 * @tparam Enabled chunk_config::use_fast_block_compression.
 *
 * @details Split out so a build WITHOUT fast block compression pays nothing for
 *          it: the false specialisation below keeps the member names, so
 *          ProxyBase still compiles, but every entry point aborts. Nothing is
 *          supposed to reach them.
 *
 * @note The `*_async` members take a context rather than reading the proxy's
 *       own fields, because on the deferred-replay path there is no live proxy
 *       -- the write is being applied long after the one that enqueued it went
 *       out of scope. @see make_proxy_with_info
 */
template <typename Derived, typename T, typename ChunkType, bool Enabled>
class ProxyBaseFastCompPolicy : public CRTPSupport<Derived>
{
  public:
    using CRTPSupport<Derived>::derived;

    /// @brief set_ptr_is_inside_chunk_list_to_compress() for the replay path,
    ///        taking the tab/chunk indices from @p ctx.
    void set_ptr_is_inside_chunk_list_to_compress_async(auto* ctx);

    /// @brief Queue the chunk @p ctx names as a compression candidate, on the
    ///        replay path.
#if CHUNK_CRTP_USE_DEDUCING_THIS
    // C++23: explicit object parameter replaces CRTPSupport::derived().
    void add_to_compression_list_async(this const auto& self, auto* ctx);
#else
    void add_to_compression_list_async(auto* ctx) const;
#endif

    /**
     * @brief Grow the per-array table of candidate-flag arrays to @p nb_tab entries.
     * @param nb_tab Number of arrays to make room for.
     */
    static void resize_tab_is_int(int nb_tab)
    {
        tab_is_inside_chunk_list_to_compress.resize(nb_tab);
    }

    /**
     * @brief Allocate and zero one array's per-chunk candidate flags.
     * @param num_tab Which array's flags to (re)allocate.
     * @param nb_chunk How many chunks it holds.
     * @note Replacing the vector frees the previous flags, so this doubles as
     *       the release path on a re-chunk.
     */
    static void resize_tab_is_int(int num_tab, int nb_chunk)
    {
        // Check builds: the row is REALLOCATED, so every flag goes back to 0.
        // A chunk still flagged here still sits in a candidate list: its entry
        // outlives the caller's chunk_list realloc, and the cleared flag lets
        // the chunk be queued twice. The callers' compression pass (resize,
        // after its deferred-op drain) must empty the lists first; this checks
        // it. Mirrors Proxy2A::resize_tab_is_int(num_tab, old, nb).
        if constexpr (build_config::any_check)
        {
            if (static_cast<std::size_t>(num_tab) < tab_is_inside_chunk_list_to_compress.size())
            {
                const auto& row = tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(num_tab)];
                for (std::size_t j = 0; j < row.size(); ++j)
                {
                    if (row[j].load(std::memory_order_relaxed) != 0) [[unlikely]]
                    {
                        CHUNK_INVARIANT_FAILED(
                            "resize_tab_is_int: tab {} chunk {} still queued for compression while its "
                            "flag row is reallocated ({} -> {} chunks)",
                            num_tab, j, row.size(), nb_chunk);
                    }
                }
            }
        }
        // Every flag 0, constructed by a team for a large array.
        tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(num_tab)] =
            parallel_storage_init::make_filled<std::atomic<char>>(static_cast<std::size_t>(nb_chunk), char{0});
    }

    /**
     * @brief Release one array's candidate flags (its teardown).
     * @details Called once the array's candidates are purged and its chunks
     *          freed. Required: a re-chunk builds the new storage under a NEW
     *          num_tab, so an old row is otherwise never replaced (one byte per
     *          old chunk leaked per re-chunk). Swapped with an empty row to free
     *          the capacity. A later resize_tab_is_int() reallocates.
     * @param num_tab The dying array's index.
     */
    static void release_tab_is_int(int num_tab) noexcept
    {
        using RowT = std::remove_reference_t<decltype(tab_is_inside_chunk_list_to_compress)>::value_type;
        if (num_tab >= 0 && static_cast<std::size_t>(num_tab) < tab_is_inside_chunk_list_to_compress.size())
        {
            RowT{}.swap(tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(num_tab)]);
        }
    }

    /// @brief Write @p v at the offset @p ctx recorded, on the replay path.
    /// @warning Reloads the chunk's value pointer from @p ctx: the one captured
    ///          when the write was enqueued may since have been reallocated.
    static void store_value_async(auto* ctx, InT<T> v) noexcept
    {
        T* const values_loc = ctx->chunk.values_load();
        elem_set(values_loc, static_cast<std::size_t>(ctx->num), static_cast<T>(v));
    }

    /// @brief Read at the offset @p ctx recorded, on the replay path.
    /// @warning Same reload caveat as store_value_async().
    /// @details Through internal::element_of(), like read_values(): a raw read
    ///          of a Chunk2A chunk compressed to its second value gives 0 (the
    ///          identifier region), and a replayed `*=` would store 0 * x.
    [[nodiscard]] static T read_values_async(auto* ctx) noexcept
    {
        return internal::element_of(ctx->chunk.values_load(), static_cast<std::size_t>(ctx->num));
    }

  protected:
    /**
     * @brief Rebuild a usable proxy from a deferred operation's context.
     * @tparam ProxyType The concrete proxy to produce.
     * @param ctx The context captured when the write was enqueued.
     * @return The rebuilt proxy, ready for the replayed write body.
     * @details Replay happens after the original proxy is gone, so the parts of
     *          it the write body needs are reconstructed here -- including the
     *          frozen chunk geometry, so the replayed write uses the geometry in
     *          force when it was enqueued rather than the current one.
     */
    template <typename ProxyType, typename Ctx>
    static ProxyType make_proxy_with_info(Ctx* ctx);

    /// @brief The one compression manager for this (T, ChunkType) pair.
    static inline constinit internal::Immortal<CompressionManager<T, ChunkType>> compression_mgr_storage_{};
    static inline CompressionManager<T, ChunkType>& compression_mgr = compression_mgr_storage_.value;

    /// @brief This proxy's entry in #tab_is_inside_chunk_list_to_compress, or
    ///        null before set_ptr_is_inside_chunk_list_to_compress() runs.
    std::atomic<char>* ptr_is_inside_chunk_list_to_compress = nullptr;

    /**
     * @brief Per-tab array of atomic flags indicating whether each chunk is marked for compression.
     * Each entry holds nb_chunk atomic<char> flags (ChunkFlagRowT), built
     * whole by make_filled (atomics are not movable).
     */

    static inline constinit internal::Immortal<std::vector<ChunkFlagRowT>>
                                              tab_is_inside_chunk_list_to_compress_storage_{};
    static inline std::vector<ChunkFlagRowT>& tab_is_inside_chunk_list_to_compress =
        tab_is_inside_chunk_list_to_compress_storage_.value;
};

/**
 * @brief ProxyBaseFastCompPolicy with fast block compression compiled out.
 *
 * @details Keeps the member NAMES so ProxyBase's `using` declarations still
 *          resolve, but the entry points abort. Reaching one means a caller
 *          took a compression path in a build that has none, which is a bug in
 *          the caller, not a case to handle.
 */
template <typename Derived, typename T, typename ChunkType>
class ProxyBaseFastCompPolicy<Derived, T, ChunkType, false> : public CRTPSupport<Derived>
{
  public:
    /// @brief No candidate flags in this configuration; nothing to size.
    /// @note Still declared: ArrayChunkBase::create_common() calls it
    ///       unconditionally, because the call site is shared with variants
    ///       that do need it.
    static void resize_tab_is_int([[maybe_unused]] int nb_tab) noexcept {}
    /// @brief No candidate flags in this configuration; nothing to size.
    static void resize_tab_is_int([[maybe_unused]] int num_tab, [[maybe_unused]] int nb_chunk) noexcept {}
    /// @brief Release one array's candidate flags, as the enabled policy does.
    /// @details Do not make this a no-op: Chunk2A's serial build has candidate
    ///          lists and sizes these rows itself (Proxy2A::resize_tab_is_int).
    ///          Otherwise a cleared array's flags stay at 1 with no list entry
    ///          behind them (one byte per chunk leaked) and the next
    ///          resize_tab_is_int() on that tab fails its "still queued" check.
    ///          v1 and Chunk2 never fill a row here: the guard makes it free.
    static void release_tab_is_int(int num_tab) noexcept
    {
        using RowT = std::remove_reference_t<decltype(tab_is_inside_chunk_list_to_compress)>::value_type;
        if (num_tab >= 0 && static_cast<std::size_t>(num_tab) < tab_is_inside_chunk_list_to_compress.size())
        {
            RowT{}.swap(tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(num_tab)]);
        }
    }

  protected:
    /// @brief Never callable in this configuration. Aborts.
    template <typename ProxyType, typename Ctx>
    static ProxyType make_proxy_with_info(Ctx* /*ctx*/)
    {
        CHUNK_INVARIANT_FAILED("make_proxy_with_info: no deferred replay in a build without fast block compression");
    }

    // Management members
    static inline constinit internal::Immortal<CompressionManager<T, ChunkType>> compression_mgr_storage_{};
    static inline CompressionManager<T, ChunkType>& compression_mgr = compression_mgr_storage_.value;
    std::atomic<char>*                              ptr_is_inside_chunk_list_to_compress;
    static inline constinit internal::Immortal<std::vector<ChunkFlagRowT>>
                                              tab_is_inside_chunk_list_to_compress_storage_{};
    static inline std::vector<ChunkFlagRowT>& tab_is_inside_chunk_list_to_compress =
        tab_is_inside_chunk_list_to_compress_storage_.value;
};

namespace internal
{
/**
 * @brief A proxy's handle on its chunk's values pointer (OpenMP builds):
 *        every read is load(ACQUIRE).
 *
 * @details The pointer publishes a buffer -- filled, then values_store(release)
 *          -- so a reader needs acquire, as ChunkBase::values_load() does. NOT
 *          seq_cst (what a plain std::atomic<T*>& converts with): no ordering
 *          here needs it -- the Dekker handshake has its own fences on the
 *          writer flags -- and on AArch64 with RCpc acquire is LDAPR, which
 *          does not wait for earlier STLR stores. NOT relaxed: nothing else
 *          orders a writer with another thread's materialisation of the buffer.
 *
 *          Implicitly built from the chunk's atomic and converted to T*, so
 *          subscript, +, ==, casts and T* arguments compile unchanged. A
 *          reference: the chunk may reallocate, so each use reloads.
 */
template <typename T>
class AcquireValuesRef
{
  public:
    // NOLINTNEXTLINE(google-explicit-constructor): built from chunk.values at every proxy construction
    AcquireValuesRef(std::atomic<T*>& ptr) noexcept : ptr_(ptr) {}

    // NOLINTNEXTLINE(google-explicit-constructor): stands for the T* it holds
    operator T*() const noexcept
    {
        return ptr_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<T*>& ptr_;
};
} // namespace internal

// ============================================================================
// ProxyBase: Base CRTP class containing all shared functionality
// ============================================================================

/**
 * @brief Base class for proxy types using CRTP pattern
 * @tparam Derived The derived proxy class (Proxy, Proxy2 or Proxy2A)
 * @tparam T The underlying value type
 * @tparam ChunkType The chunk type (Chunk<T>, Chunk2<T> or Chunk2A<T>)
 *
 * Contains all common operators, compression logic, and thread-safety mechanisms.
 *
 * A proxy stands for ONE element of an array and behaves like a value: it
 * converts to T by reading the element (operator T() -> read_pure(), which
 * first synchronises any deferred write to it), and every write goes through
 * it. It is a handle, not a copy:
 *   - it holds references into the array's chunk storage, so it must not
 *     outlive an operation that rebuilds that storage -- in particular a
 *     re-chunk, explicit or decided by a RechunkDecider (full list on the
 *     array's non-const operator[]);
 *   - `auto v = a[i];` keeps the proxy, which then follows the element;
 *     `T v = a[i];` takes the value.
 * The compound operators (+= -= *= /= %= &= |= ^= <<= >>=) and the prefix
 * ++/-- return the proxy by value: discarding it, as `a[i] += x;` does, keeps
 * the operation fully deferrable, while `T y = (a[i] += x);` reads the result.
 * The postfix ++/-- return the old value as a T.
 */
template <typename Derived, typename T, typename ChunkType>
// copy rebinds to the same element; the derived proxies
// define what assignment means, so no assignment or move is declared here.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class ProxyBase : public ProxyOperators<Derived, T>,
                  public ProxyBaseFastCompPolicy<Derived, T, ChunkType, chunk_config::use_fast_block_compression>

{
  protected:
    template <typename U, typename D, typename C, bool E>
    friend class ProxyBaseFastCompPolicy;

    // ========================================================================
    // Common type aliases
    // ========================================================================
    // OpenMP: every read of the chunk's pointer is load(acquire), see
    // internal::AcquireValuesRef.
    using AtomicPtrRefT = std::conditional_t<build_config::use_openmp, internal::AcquireValuesRef<T>, T*&>;

    /// @brief Check builds: the array's storage generation, and its value when
    ///        operator[] built this proxy (ArrayChunkBase::storage_generation_).
    ///        Empty elsewhere. gen stays null on the proxies the library builds
    ///        for itself (deferred replays): those are never stale.
    struct StaleGuardOn
    {
        const unsigned int* gen  = nullptr;
        unsigned int        seen = 0;
    };
    struct StaleGuardOff
    {
    };
    [[no_unique_address]] mutable std::conditional_t<build_config::any_check, StaleGuardOn, StaleGuardOff>
        stale_guard_{};

    /// @brief Check builds: abort when the array rebuilt its storage after
    ///        operator[] built this proxy -- the proxy then points into
    ///        swapped-out or released chunks. Called first by every read and
    ///        write entry (read_pure, operator T, assign_impl, the compound and
    ///        in-place funnels). Nothing in other builds.
    void check_not_stale() const noexcept
    {
        if constexpr (build_config::any_check)
        {
            if (stale_guard_.gen != nullptr && *stale_guard_.gen != stale_guard_.seen) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED(
                    "write proxy of array {} (chunk {}, offset {}) used after the array rebuilt its "
                    "storage (resize, reserve, shrink_to_fit, clear, assign, swap, an assignment or a "
                    "re-chunk): keep a value with `T v = a[i];`, not `auto v = a[i];`",
                    ptr_on_array, num_chunk, num);
            }
        }
    }

    using CompressionManagerType = CompressionManager<T, ChunkType>;

    /// @brief Record which array this proxy points into. @see get_num_tab
    void set_num_tab(int nt) noexcept
    {
        num_tab = nt;
    }

    // ========================================================================
    // Common chunk state queries
    // ========================================================================

    /// @brief Is the chunk compressed to its single first value?
    [[nodiscard]] bool is_cst_val1() const noexcept
    {
        return chunk.is_cst_val1();
    }

    /// @brief is_cst_val1() judged from an already-loaded value pointer.
    /// @param values_loc The pointer the caller loaded; saves a reload on a path
    ///        that has one in hand, and keeps the whole test on ONE snapshot.
    [[nodiscard]] bool is_cst_val1(const T* values_loc) const noexcept
    {
        return chunk.is_cst_val1(values_loc);
    }

    /// @brief Complement of is_cst_val1().
    [[nodiscard]] bool is_not_cst_val1() const noexcept
    {
        return !chunk.is_cst_val1();
    }

    /// @brief Would writing @p v leave the chunk compressed exactly as it is?
    /// @details The write fast path's early-out: the chunk holds only its first
    ///          value and @p v IS that value, so there is nothing to do and no
    ///          reason to materialise the chunk.
    ///
    ///          Never while the chunk has a deferred op pending
    ///          (pending_any()): that op is OLDER than this write in program
    ///          order and would replay after it, so the write must reach the
    ///          engine, whose WAW guard retires the op before the store;
    ///          otherwise the write is silently lost. pending_any() is one
    ///          acquire load of the chunk header, constant-false in a serial build.
    [[nodiscard]] bool is_cst_val1_and_v_equal_first_value(InT<T> v) const noexcept
    {
        return v == static_cast<T>(ChunkType::first_value) && is_cst_val1() && !chunk.pending_any();
    }

    /// @brief is_cst_val1_and_v_equal_first_value() from a loaded value pointer.
    [[nodiscard]] bool is_cst_val1_and_v_equal_first_value(const T* values_loc, InT<T> v) const noexcept
    {
        return v == static_cast<T>(ChunkType::first_value) && is_cst_val1(values_loc) && !chunk.pending_any();
    }

    /// @brief Point this proxy at an already-known candidate flag.
    void set_ptr_is_inside_chunk_list_to_compress(std::atomic<char>* pt) noexcept
    {
        ptr_is_inside_chunk_list_to_compress = pt;
    }
    /**
     * @brief Set up pointer to this chunk's compression tracking flag
     * @pre num_tab and num_chunk must be valid
     * @post ptr_is_inside_chunk_list_to_compress points to correct atomic flag
     */
    void set_ptr_is_inside_chunk_list_to_compress() noexcept
    {
        ptr_is_inside_chunk_list_to_compress =
            &FastCompPolicy::tab_is_inside_chunk_list_to_compress[num_tab][num_chunk];
    }

    /// @brief add_to_compression_list() on the calling thread's own list.
    void add_to_compression_list() const;

    // ========================================================================
    // Common I/O operations
    //

    /// @brief This element, read raw: `values[num]`, or its bit for Bit
    ///        (element_access.h). No sync, no check.
    [[nodiscard]] T raw_elem() const noexcept
    {
        // Only the serial counter mode reads through this: no pass runs
        // concurrently there. A build with in-region passes must read through
        // the write session, element_at, or read_values_outside_session.
        static_assert(chunk_config::use_compression_with_counter || sizeof(T) == 0,
                      "raw_elem(): serial counter mode only -- no concurrent pass may release the buffer");
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        return elem_get(static_cast<const T*>(values), static_cast<std::size_t>(num));
    }

    /// @brief This element, written raw: `values[num] = v`, or an atomic RMW
    ///        of its word for Bit. No sync, no check.
    void raw_elem_set(InT<T> v) const noexcept
    {
        elem_set(static_cast<T*>(values), static_cast<std::size_t>(num), static_cast<T>(v));
    }

    /// @brief Write @p v straight through #values, with no synchronisation.
    /// @pre The chunk is materialised and the caller holds the write path.
    void store_value(InT<T> v) const noexcept
    {
        store_value_at(values, v);
    }

    /// @brief store_value() through a pointer the caller has already loaded.
    /// @param vals The chunk's values pointer, loaded inside the write session
    ///        (after the materialisation, if there was one); saves a reload.
    /// @param v The value to store.
    /// @pre As store_value().
    void store_value_at(T* const vals, InT<T> v) const noexcept
    {
        // The chunk must not be COMMITTED to a compressor here: a writer on a
        // marked chunk is sent to the slow path by must_defer_write(), and a
        // compressor commits its claims only once its window is shut -- no
        // writer flagged on a marked chunk. Do not test "merely marked" (1):
        // the compressor may have marked it after this writer's check #2 and
        // be waiting on the flag (false positive). If this fires, the
        // compressor passed its window with this writer's flag up: suspect the
        // ordering of the gate's read, not the propagation of the mark (this
        // write then races scan_constants()/read_value_scan()).
        if constexpr (build_config::any_check && chunk_config::use_chunk_steal_mark)
        {
            if (chunk.steal_committed()) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("store_value on a chunk committed to a compressor (num={})", num);
            }
        }
        // Chunk2A: the marking filter of another writer may be reading this
        // element -- relaxed atomic store, see relaxed_element.h.
        if constexpr (internal::relaxed_element_stores_v<ChunkType, T>)
        {
            internal::store_element_relaxed(vals + num, static_cast<T>(v));
        }
        else
        {
            // Bit: an atomic RMW of the word (element_access.h); values[num] otherwise.
            elem_set(vals, static_cast<std::size_t>(num), static_cast<T>(v));
        }
    }

    /// @brief Raw read for the WRITE path only. Deliberately does not sync.
    /// @warning Not an ordinary read -- use read_pure(). The reason it must not
    ///          sync is in the body comment.
    [[nodiscard]] T read_values() const noexcept
    {
        // WRITE-PATH helper ONLY (v_old capture + mark_for_compression new value
        // in execute_simple_operation / arithmetic_assign). It must NOT sync:
        // syncing re-flushes from inside a write (begin_write held) and, when
        // pending_ops_>0, wait_for_availability's replay reallocates `values`
        // under the in-flight writer. Same-offset visibility on the write path
        // is handled by the WAW erase before the write body; pure reads use
        // read_pure(). Not raw for a Chunk2A second constant (see
        // internal::element_of()): the compression decision must see the
        // constant, not 0.
        return read_values_at(values);
    }
    /// @brief read_values() through a pointer the caller has already loaded.
    /// @param vals The chunk's values pointer, loaded inside the write session;
    ///        saves the reload when the write path has one in hand.
    [[nodiscard]] T read_values_at(const T* const vals) const noexcept
    {
        // TEST KNOB (test_hooks.h), as in ChunkBase::element_at: widens
        // the load -> dereference window. Compiled out by default.
        test_hooks::maybe_read_delay();
        return internal::element_of(vals, static_cast<std::size_t>(num));
    }
    /// @brief read_values() for a pre-check made BEFORE the write session
    ///        (the `0 * x` shortcut of the compound operations): inside a
    ///        read_reclaim::ReadSection.
    /// @details Outside the session a pass in the region may compress the chunk
    ///          and release its buffer between the pointer load and the element
    ///          load; the next materialisation reuses and overwrites it (zeros,
    ///          for a first constant), and the shortcut would wrongly drop the
    ///          operation. The section makes that pass wait for this read
    ///          before it releases anything, as for element_at.
    [[nodiscard]] T read_values_outside_session() const noexcept
    {
        const read_reclaim::ReadSection section;
        return read_values();
    }
    /// @brief Read through the chunk WITHOUT synchronising first.
    /// @warning Returns a stale value if a deferred write on this offset has
    ///          not been applied. Use read_pure() unless you know otherwise.
    [[nodiscard]] T load_value() const noexcept
    {
        return chunk.read_value(num);
    }

    /**
     * @brief Run @p write_body on the fast path, or defer it.
     * @tparam WriteCallable The write itself.
     * @tparam LbdInfo Callable arming whatever metadata the write needs.
     * @return TRUE if the write was performed, FALSE if it was deferred and
     *         will be replayed later.
     * @note @p lbd_info runs BEFORE the body, and only when the metadata is
     *       actually needed -- it is a no-op without fast block compression.
     * @note Force-inlined, like the const operator[] (array_chunk_base.h):
     *       otherwise GCC emits one out-of-line copy per (proxy, type, write
     *       lambda) and every write pays a call. Only this body is forced --
     *       the slow-path callees (wait_out_compressor, run_slow_path,
     *       flush_deferred, ...) stay calls; do NOT turn this into
     *       CHUNK_FLATTEN, which would copy them into every write site.
     */
    template <typename WriteCallable, typename LbdInfo>
    [[nodiscard]] CHUNK_FORCE_INLINE bool write_with_optional_lock_impl(WriteCallable&&          write_body,
                                                                        [[maybe_unused]] LbdInfo lbd_info) noexcept;

    /**
     * @brief Record a write to be applied later, when the fast path refuses it.
     * @param ctx Captures everything the replay needs -- offset, chunk indices,
     *        and the frozen geometry; already pool-allocated by the caller.
     * @param body The write, run at replay time against a proxy rebuilt by
     *        make_proxy_with_info().
     */
    template <typename CtxType, typename LambdaBody>
    void enqueue_deferred_write(CtxType* ctx, LambdaBody&& body) const;

    /**
     * @brief Apply @p operation to the element: load, compute in
     *        common_type_t<T, U>, store -- inside the write protocol
     *        (write_with_optional_lock_info), deferred when the fast path
     *        refuses. Not a hardware atomic read-modify-write: there is no
     *        compare-exchange loop, and two threads updating the same element
     *        race as they would on a plain T.
     * @param operation Operation to apply.
     * @param v         Operand.
     * @param v_old     Unused by the body (kept for the callers' signature; the
     *                  bodies reload the current value themselves).
     */
    template <typename U, typename Op>
        requires std::is_invocable_r_v<T, Op, T, U>
    void apply_atomic_op(Op operation, const U& v, InT<T> v_old);

    /**
     * @brief Generic helper for compound arithmetic operations
     *
     * Executes an arithmetic operation with synchronization and optional statistics.
     * Used to implement += and -=.
     *
     * @tparam U Type of the value to apply
     * @tparam Op Operation type (callable: T(T, U))
     * @param v Value to use in the operation
     * @param operation Binary function applying the operation (e.g., a + b)
     * @note Returns nothing: the operators return the proxy, whose conversion
     *       to T reads the resulting value (synchronised). On the deferred
     *       path the result does not exist yet on return, so nothing here
     *       could report it.
     * @note Thread-safe via write_with_optional_lock_info()
     * @note Increments CHUNK_STATS statistics if enabled
     */
    template <typename U, typename Op>
    void arithmetic_assign(InT<U> v, Op operation) noexcept; // InT: see assign_impl in the proxies

    /**
     * @brief Has a thread's candidate list grown past the point where it must be drained?
     *
     * The single definition of the "list became dangerous" threshold, shared by
     * the trigger_compression_if_needed() of Proxy, Proxy2 and Proxy2A. Keep it
     * that way: three copies of the comparison drift apart.
     *
     * @param sz  Current candidate count for the thread.
     * @param szm Maximum tolerated before the list is flagged dangerous.
     * @return True when the list must be flagged.
     */
    static bool should_trigger_compression(int sz, int szm) noexcept
    {
        return sz > szm;
    }

    // ========================================================================
    // Common member variables
    // ========================================================================
    /// @brief Reference to the chunk's value pointer; under OpenMP an
    ///        AcquireValuesRef, read with acquire on every use.
    /// @warning A REFERENCE, captured at construction. The chunk may reallocate
    ///          under it, so a write path must reload through it, never cache it.
    AtomicPtrRefT values;

    const int  num;   ///< Offset of this element WITHIN its chunk.
    ChunkType& chunk; ///< The chunk that owns the element.

    mutable const void* ptr_on_array; ///< The array the chunk belongs to.
    mutable int         num_chunk;    ///< Index of the chunk within that array.
    mutable int         num_tab = -1; ///< Index of the array in the registry; -1 until armed.

    // Frozen chunk geometry for the deferred-replay path.
    // 0 means "not armed" -> chunk_size() falls back to the live ptr_on_array
    // read (the direct write path, array alive). make_proxy_with_info arms this
    // from ctx->chunk_size_frozen on replay, so chunk_size() there returns the
    // enqueue-time value WITHOUT dereferencing a possibly-dead ptr_on_array.
    mutable unsigned int chunk_size_frozen_ = 0;
    /// @brief Affinity key captured with chunk_size_frozen_ for the same reason:
    ///        on a deferred replay ptr_on_array may be dead. 0 until frozen.
    mutable unsigned int alloc_key_frozen_ = 0;

  public:
    // Required for template concepts to recognize this as a Proxy
    using is_shrink_proxy_tag = void;
    using value_type          = T;

    using FastCompPolicy = ProxyBaseFastCompPolicy<Derived, T, ChunkType, chunk_config::use_fast_block_compression>;

    using FastCompPolicy::compression_mgr;
    using FastCompPolicy::derived;
    using FastCompPolicy::ptr_is_inside_chunk_list_to_compress;

    /**
     * @brief Modulo struct for compound modulo operation
     * @details Wrapper for % operator to fit execute_arithmetic_compound_operation interface
     */

    struct modulo
    {
        /**
         * @brief Apply modulo operation
         * @param lhs Dividend
         * @param rhs Divisor
         * @return lhs % rhs
         */
        constexpr T operator()(T lhs, T rhs) const noexcept
        {
            return lhs % rhs;
        }
    };

    /// @brief The operand kinds the Bit operators take: bool, Bit, or a proxy
    ///        of either.
    template <typename U>
    static constexpr bool bit_operand_v = std::is_same_v<BaseT<U>, bool> || is_packed_bit_v<BaseT<U>>;

    /// @brief Bind a proxy to one element.
    /// @note Leaves #ptr_on_array, #num_chunk and #num_tab unset --
    ///       set_this_and_num_chunk() and set_num_tab() finish the job, and the
    ///       compression paths need them.
    // ptr_on_array and num_chunk are set before use (see the @note); left
    // unset here on purpose, this constructor runs on every element access.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    explicit ProxyBase(AtomicPtrRefT values_, int num_, ChunkType& chunk_) noexcept
        : values(values_), num(num_), chunk(chunk_)
    {
    }

    // ========================================================================
    // DELETED OPERATORS
    // ========================================================================

    /// @brief Deleted: proxy-to-proxy assignment would be ambiguous -- rebind
    ///        the proxy, or copy the value? The derived classes define what
    ///        `a[i] = b[j]` means.
    const Derived& operator=(const Derived& from) = delete;

    /// @brief Deleted: there is no stable address to take. The chunk may be
    ///        compressed, or reallocate under the proxy.
    T* operator&() = delete;

    /// @brief Deleted, as the non-const overload: no stable address to take.
    const T* operator&() const = delete;

    /// @brief Copying a proxy is fine: it rebinds to the same element.
    ProxyBase(const ProxyBase& other) = default;

    // ========================================================================
    // Common getters
    // ========================================================================

    /// @brief Index of this element's chunk within its array.
    [[nodiscard]] int get_num_chunk() const noexcept
    {
        return num_chunk;
    }

    /**
     * @brief Elements per chunk, as this proxy sees them.
     *
     * @details A compile-time constant with a static chunk size; with a dynamic
     *          one it reads the array's runtime geometry through #ptr_on_array
     *          -- unless set_chunk_size_frozen() armed a value, which the replay
     *          path does precisely so it need not touch a possibly-dead
     *          #ptr_on_array.
     *
     * @note Use this rather than the process-wide geometry on any write path:
     *       during a re-chunk the two disagree.
     */
    [[nodiscard]] unsigned int chunk_size() const noexcept;
    /// @brief The owning array's slab-affinity key (0 = shared pool); frozen
    ///        twin on the replay path like chunk_size().
    [[nodiscard]] unsigned int alloc_key() const noexcept;

    /// @brief Offset of this element within its chunk.
    [[nodiscard]] int get_num() const noexcept
    {
        return num;
    }

    /// @brief Index of this element's array in the proxy registry.
    [[nodiscard]] int get_num_tab() const noexcept
    {
        return num_tab;
    }

    /// @brief The chunk that owns this element.
    [[nodiscard]] ChunkType& get_chunk() const noexcept
    {
        return chunk;
    }

    /// @brief Queue this chunk as a compression candidate in @p vec_tuple.
    void add_to_compression_list([[maybe_unused]] auto& vec_tuple) const;

    /// @brief The ordinary read: synchronise this offset, then read it live.
    /// @details The only place the read-side synchronisation cost is paid, and
    ///          the entry point behind `operator T()`.
    [[nodiscard]] T read_pure() const noexcept
    {
        check_not_stale();
        chunk.reader_sync(num);
        return chunk.read_value(num); // read_value accounts the read in stats
    }
    // ========================================================================
    // Common metadata management
    // ========================================================================

    /// @brief Check builds: arm the stale-proxy check with the array's storage
    ///        generation (see check_not_stale). Nothing in other builds.
    /// @param gen the array's ArrayChunkBase::storage_generation_
    void arm_storage_generation([[maybe_unused]] const unsigned int* gen) const noexcept
    {
        if constexpr (build_config::any_check)
        {
            stale_guard_.gen  = gen;
            stale_guard_.seen = *gen;
        }
    }

    /// @brief Arm the proxy with its array and chunk index; the table number
    ///        is left unresolved (set_num_tab() looks it up).
    /// @param array_ptr The owning array (its registry key).
    /// @param num_chunk_ Chunk index within that array.
    void set_this_and_num_chunk(const void* array_ptr, int num_chunk_) const noexcept
    {
        ptr_on_array = array_ptr;
        num_chunk    = num_chunk_;
        num_tab      = -1; // not armed: set_num_tab() looks the registry up
    }

    /// @brief Same, with the array's own table number, so the write path
    ///        needs no registry lookup.
    /// @param array_ptr  pointer to the array
    /// @param num_chunk_ chunk index
    /// @param num_tab_   the array's num_tab (ArrayChunkBase::num_tab), the
    ///                   value it registered under its address
    /// @note Avoids the registry lookup (share_info_array->get_num_tab, a
    ///       linear scan or hash probe) on every write, a measurable cost. The
    ///       value is checked against the registry under any_check
    ///       (Proxy2A::set_num_tab).
    void set_this_and_num_chunk(const void* array_ptr, int num_chunk_, int num_tab_) const noexcept
    {
        ptr_on_array = array_ptr;
        num_chunk    = num_chunk_;
        num_tab      = num_tab_;
    }

    /// @brief Compare the element's value with @p v. Reads through read_pure().
    template <typename U>
    [[nodiscard]] bool operator==(const U& v) const noexcept
    {
        return read_pure() == v;
    }

    /// @brief Compare with a @p v already of the element type. Also serves
    ///        `!=` (C++20 rewrites it as `!(p == v)`).
    [[nodiscard]] bool operator==(T v) const noexcept
    {
        return read_pure() == v;
    }

    /// @brief Is the element's value falsy?
    [[nodiscard]] bool operator!() const noexcept
    {
        return !read_pure();
    }

    /// @brief The element's value, negated. Does not modify it.
    [[nodiscard]] T operator-() const noexcept
    {
        return -read_pure();
    }

    /// @brief The element's value divided by @p v. Does not modify it.
    template <typename U>
    [[nodiscard]] T operator/(const U& v) const noexcept
        requires(!std::is_same_v<U, std::complex<T>>)
    {
        return read_pure() / static_cast<T>(v);
    }

    /// @brief Read the element. This is what makes a proxy behave like a value.
    /// @note Goes through read_pure(), so it pays the read-side synchronisation.
    [[nodiscard]] operator T() const noexcept
    {
        return read_pure();
    }

    /// @brief ArrayChunkBits: read the element straight as a bool. Through
    ///        operator T() it would be two user conversions (proxy -> Bit ->
    ///        bool), which C++ does not chain, so `bool b = a[i]` needs this one.
    [[nodiscard]] operator bool() const noexcept
        requires is_packed_bit_v<T>
    {
        return static_cast<bool>(read_pure());
    }

    /**
     * @brief Compound addition assignment -- clean types (e.g. float += float,
     *        float += int).
     * @tparam U Type of operand (implicit conversion to T)
     * @param v Value to add
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires IsCleanCompoundOp<T, U> && ElementAddable<T>
    Derived operator+=(const U& v) noexcept
    {
        arithmetic_assign<T>(static_cast<T>(v), std::plus<T>{});
        return derived();
    }

    /** @brief Compound addition — type mismatch (e.g. float += double). Precision warning. */
    template <typename U>
        requires(!IsCleanCompoundOp<T, U>) && std::is_convertible_v<U, T> && ElementAddable<T>
    [[deprecated("HPC WARNING: Type mismatch in +=. Possible precision loss. Use 'f' suffix for floats.")]]
    Derived operator+=(const U& v) noexcept
    {
        arithmetic_assign<T>(static_cast<T>(v), std::plus<T>{});
        return derived();
    }

    /**
     * @brief Compound subtraction operator -- clean types.
     *
     * @tparam U Type of the value to subtract (must be compatible with T)
     * @param v Value to subtract
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     * @note noexcept guaranteed
     * @note Increments CHUNK_STATS if enabled
     */
    template <typename U>
        requires IsCleanCompoundOp<T, U> && ElementSubtractable<T>
    Derived operator-=(const U& v) noexcept
    {
        arithmetic_assign<T>(static_cast<T>(v), std::minus<T>{});
        return derived();
    }

    /** @brief Compound subtraction — type mismatch. */

    template <typename U>
        requires(!IsCleanCompoundOp<T, U>) && std::is_constructible_v<T, U> && ElementSubtractable<T>
    [[deprecated("HPC WARNING: Precision mismatch.")]]
    Derived operator-=(const U& v) noexcept
    {
        arithmetic_assign<T>(static_cast<T>(v), std::minus<T>{});
        return derived();
    }

    /**
     * @brief Compound bitwise AND operator
     *
     * @tparam U Type of the value (must be integral)
     * @param v Value to use for the AND operation
     * @return This proxy (see operator+=).
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     */

    template <typename U>
    Derived operator&=(const U& v) noexcept
        requires std::is_integral_v<T> && std::is_integral_v<BaseT<U>>
    {
        bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs & rhs; });
        return derived();
    }

    /**
     * @brief Compound bitwise OR operator
     *
     * @tparam U Type of the value (must be integral)
     * @param v Value to use for the OR operation
     * @return This proxy (see operator+=).
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     */

    template <typename U>
    Derived operator|=(const U& v) noexcept
        requires std::is_integral_v<T> && std::is_integral_v<BaseT<U>>
    {
        bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs | rhs; });
        return derived();
    }

    /**
     * @brief Compound bitwise XOR operator
     *
     * @tparam U Type of the value (must be integral)
     * @param v Value to use for the XOR operation
     * @return This proxy (see operator+=).
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     */

    template <typename U>
    Derived operator^=(const U& v) noexcept
        requires std::is_integral_v<T> && std::is_integral_v<BaseT<U>>
    {
        bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs ^ rhs; });
        return derived();
    }

    // ------------------------------------------------------------------------
    // ArrayChunkBits (T = Bit, bit.h): |= &= ^= with a bool operand.
    //
    // Bit is not integral, so the three above do not apply. A logical
    // operation needs no read-modify-write when the operand decides the result:
    // `|= true` is `= true`, `&= false` is `= false`, and the other operand
    // leaves the element as it is -- a plain write through the proxy's
    // operator=, with its whole protocol (marking, deferral), and no write at
    // all when nothing changes. Only `^= true` reads: the element, then its
    // negation written back -- as for a plain bool, two threads updating the
    // SAME element race; different elements of one word do not.
    // ------------------------------------------------------------------------

    /// @brief `a[i] |= b` for ArrayChunkBits: sets the element when @p v is true.
    template <typename U>
    Derived operator|=(const U& v) noexcept
        requires is_packed_bit_v<T> && bit_operand_v<U>
    {
        if (static_cast<bool>(v))
        {
            derived() = true;
        }
        return derived();
    }

    /// @brief `a[i] &= b` for ArrayChunkBits: clears the element when @p v is false.
    template <typename U>
    Derived operator&=(const U& v) noexcept
        requires is_packed_bit_v<T> && bit_operand_v<U>
    {
        if (!static_cast<bool>(v))
        {
            derived() = false;
        }
        return derived();
    }

    /// @brief `a[i] ^= b` for ArrayChunkBits: flips the element when @p v is true.
    template <typename U>
    Derived operator^=(const U& v) noexcept
        requires is_packed_bit_v<T> && bit_operand_v<U>
    {
        if (static_cast<bool>(v))
        {
            const bool cur = static_cast<bool>(read_pure()); // synchronised read, as operator bool()
            derived()      = !cur;
        }
        return derived();
    }

    /**
     * @brief Compound left shift operator
     *
     * @tparam U Type of the shift amount (must be integral)
     * @param v Number of positions to shift
     * @return This proxy (see operator+=).
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     */

    /** @brief Bitwise shift left — clean types (Integral <<= Integral). */
    template <typename U>
        requires IsCleanShiftOp<T, U>
    Derived operator<<=(const U& v) noexcept
    {
        bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs << rhs; });
        return derived();
    }

    /** @brief Bitwise shift left — trap for non-integral shift amount. */
    template <typename U>
        requires(!IsCleanShiftOp<T, U>) && std::is_integral_v<T>
    [[deprecated("HPC WARNING: Shift amount must be an integer to avoid precision issues.")]]
    Derived operator<<=(const U& v) noexcept
    {
        bitwise_assign<int>(static_cast<int>(v), [](auto lhs, auto rhs) { return lhs << rhs; });
        return derived();
    }

    /**
     * @brief Compound right shift operator
     *
     * @tparam U Type of the shift amount (must be integral)
     * @param v Number of positions to shift
     * @return This proxy (see operator+=).
     * @note Requires T and U to be integral types
     * @note noexcept guaranteed
     * @note Divides by 2^v for unsigned types, implementation-defined for signed
     */
    template <typename U>
        requires IsCleanShiftOp<T, U>
    Derived operator>>=(const U& v) noexcept
    {
        bitwise_assign<U>(v, [](auto lhs, auto rhs) { return lhs >> rhs; });
        return derived();
    }

    /** @brief Bitwise shift right — trap. */
    template <typename U>
        requires(!IsCleanShiftOp<T, U>) && std::is_integral_v<T>
    [[deprecated("HPC WARNING: Shift amount must be an integer.")]]
    Derived operator>>=(const U& v) noexcept
    {
        bitwise_assign<int>(static_cast<int>(v), [](auto lhs, auto rhs) { return lhs >> rhs; });
        return derived();
    }

    /// @brief Bitwise complement of the element's value (read through read_pure()).
    [[nodiscard]] T operator~() const noexcept
        requires std::is_integral_v<T>
    {
        return ~read_pure();
    }

    /**
     * @brief Pre-increment operator
     * @return This proxy (see operator+=).
     * @details Implements: ++proxy
     * Returns the value after increment.
     */
    Derived operator++() noexcept
        requires ElementIncrementable<T>
    {
        const auto inc_operation = [this] { derived().execute_inplace_operation([](T& elem) { ++elem; }); };

        const auto inc_operation_async = [](auto* ctx, auto& proxy)
        { proxy.execute_inplace_operation_async(ctx, [](T& elem) { ++elem; }); };

        apply_inplace_operation(inc_operation, inc_operation_async);

        // Return the proxy, not a value. Never read `values[num]` here: `values`
        // is the ctor-time snapshot, and the call above may have materialised
        // the chunk (snapshot stale, possibly the cst sentinel) or DEFERRED the
        // op (cell not yet updated, another rank's replay may be writing it).
        // The returned proxy's operator T() is read_pure(): `y = ++a[i]` gets
        // the right value, and the synchronised read is only paid when used.
        return derived();
    }

    /**
     * @brief Pre-decrement operator
     * @return This proxy (see operator+=).
     * @details Implements: --proxy
     * Returns the value after decrement.
     */
    Derived operator--() noexcept
        requires ElementIncrementable<T>
    {
        const auto dec_operation = [this] { derived().execute_inplace_operation([](T& elem) { --elem; }); };

        const auto dec_operation_async = [](auto* ctx, auto& proxy)
        { proxy.execute_inplace_operation_async(ctx, [](T& elem) { --elem; }); };

        apply_inplace_operation(dec_operation, dec_operation_async);

        // Return the proxy, not a value: see operator++().
        return derived();
    }

    /**
     * @brief Post-increment operator
     * @return Value before increment
     *
     * @details Implements: proxy++
     * Returns the original value before increment.
     */
    T operator++(int) noexcept
        requires ElementIncrementable<T>
    {
        T temp(*this);

        const auto inc_operation = [this] { derived().execute_inplace_operation([](T& elem) { elem++; }); };

        const auto inc_operation_async = [](auto* ctx, auto& proxy)
        { proxy.execute_inplace_operation_async(ctx, [](T& elem) { elem++; }); };

        apply_inplace_operation(inc_operation, inc_operation_async);
        return temp;
    }

    /**
     * @brief Post-decrement operator
     * @return Value before decrement
     *
     * @details Implements: proxy--
     * Returns the original value before decrement.
     */
    T operator--(int) noexcept
        requires ElementIncrementable<T>
    {
        T          temp(*this);
        const auto dec_operation = [this] { derived().execute_inplace_operation([](T& elem) { elem--; }); };

        const auto dec_operation_async = [](auto* ctx, auto& proxy)
        { proxy.execute_inplace_operation_async(ctx, [](T& elem) { elem--; }); };

        apply_inplace_operation(dec_operation, dec_operation_async);
        return temp;
    }

    /**
     * @brief Set the candidate-list flag of chunk @p num_chunkl of array @p num_tabl.
     * @param num_tabl Array (table) number.
     * @param num_chunkl Chunk index within that array.
     * @param v New flag value (0: not queued).
     */
    static void set_tab_is_inside_chunk_list_to_compress(int num_tabl, int num_chunkl, char v) noexcept
    {
        FastCompPolicy::tab_is_inside_chunk_list_to_compress[num_tabl][num_chunkl] = v;
    }

    /// @brief The whole candidate-flag row for array @p num_tab.
    static ChunkFlagRowT& get_tab_is_inside_list_to_compress(int num_tab) noexcept
    {
        return FastCompPolicy::tab_is_inside_chunk_list_to_compress[num_tab];
    }

    /// @brief Clear one chunk's candidate flag, so it may be queued again.
    /// @note Release store: whatever made the chunk a candidate must be visible
    ///       to the thread that next reads the flag.
    static void reset_tab_inside_list_to_compress(int num_tab, int num_chunk) noexcept
    {
        FastCompPolicy::tab_is_inside_chunk_list_to_compress[num_tab][num_chunk].store(0, std::memory_order_release);
    }

  private:
    /**
     * @brief Executes a write operation without parameters.
     * @tparam WriteFn Callable to execute.
     * @param write_fn Function to execute.
     */
    template <typename WriteFn>
    [[nodiscard]] CHUNK_FORCE_INLINE bool write_with_optional_lock_info(WriteFn write_fn)
    {
        const auto lbd_info = [this]
        {
            if constexpr (chunk_config::use_fast_block_compression)
            {
                derived().set_info_on_chunk();
            }
        };

        return write_with_optional_lock_impl([&] { write_fn(); }, lbd_info);
    }

    /**
     * @brief Deferred-path tail: a dangerous rank still drives its pass.
     *
     * A write that must be deferred (a compressor is in flight) leaves without
     * further ado -- unless this rank's candidate list is flagged dangerous, in
     * which case it runs perform_compression() as the fast path would have
     * after its store: the flush of the rank's own deferred ops, then a
     * start() that waits for the in-flight compressor rather than yielding
     * to a flush of its own making. The other ranks just leave: a pass of
     * theirs would flush -- and announce -- the single op it had just enqueued.
     *
     * @param thr Calling thread's rank.
     */
    void drive_compression_if_dangerous(int thr) noexcept
    {
        if (CompressionManagerType::is_dangerous_list(thr)) [[unlikely]]
        {
            compression_mgr.run_slow_path([&] { derived().perform_compression(); });
        }
    }

    /// @brief Pin the chunk geometry a deferred write must be replayed with.
    /// @param sz Element count per chunk at enqueue time; 0 means "not armed",
    ///        and chunk_size() then reads the live geometry instead.
    /// @param key Allocator affinity key at enqueue time (0: shared pool).
    /// @details Without this, replay would read the geometry through
    ///          #ptr_on_array -- which a re-chunk may have freed by then.
    void set_chunk_size_frozen(unsigned int sz, unsigned int key = 0) const noexcept
    {
        chunk_size_frozen_ = sz;
        alloc_key_frozen_  = key;
    }

    /// @brief Record the chunk index alone. @see set_this_and_num_chunk
    void set_num_chunk(int num_chunk_) const noexcept
    {
        num_chunk = num_chunk_;
    }

    /// @brief enqueue_deferred_write() for a compound arithmetic operation,
    ///        carrying @p v and @p operation into the context.
    template <typename U, typename Op, typename AsyncBody>
    void enqueue_async_arithmetic(const U& v, Op operation, AsyncBody&& async_body);

    /// @brief Run @p lambda as a simple write, through the derived proxy.
    template <typename Lambda>
    void execute_operation(Lambda&& lambda)
    {
        derived().execute_simple_operation(std::forward<Lambda>(lambda));
    }

    /// @brief execute_operation() on the replay path, against @p ctx.
    template <typename Lambda>
    void execute_operation_async(auto* ctx, Lambda&& lambda)
    {
        derived().execute_simple_operation_async(ctx, std::forward<Lambda>(lambda));
    }

    /**
     * @brief Engine of every bitwise compound operator (&= |= ^= <<= >>=).
     *
     * @tparam U Type of the value to apply (must be integral)
     * @tparam Op Operation type (callable: T(T, T))
     * @param v Value to use in the operation
     * @param operation Function applying the operation
     * @note Returns nothing: the operators return the proxy, whose conversion
     *       to T reads the resulting value (synchronised). On the deferred
     *       path the result does not exist yet on return, so nothing here
     *       could report it.
     * @note Thread-safe via write_with_optional_lock_info()
     * @note Increments CHUNK_STATS statistics if enabled, for every operator
     * @note Requires T and U to be integral types
     */
    template <typename U, typename Op>
    void bitwise_assign(InT<U> v, Op operation) noexcept
        requires std::is_integral_v<T> && std::is_integral_v<U>;

    /**
     * @brief Helper to execute an increment/decrement operation with optional async handling
     *
     * @tparam SyncOp  Callable applying the operation in place, now.
     * @tparam AsyncOp Callable `(ctx, proxy)` applying it later, on replay.
     * @param sync_op  Body run on the fast path.
     * @param async_op Body enqueued when the fast path refuses the write.
     * @note Materialises any foreign deferred write on this offset first: an
     *       in-place operation reads the current value, so a stale read here
     *       would clobber that write.
     */
    template <typename SyncOp, typename AsyncOp>
    void apply_inplace_operation(SyncOp sync_op, AsyncOp async_op) noexcept;
};

} // namespace shrink_array

#include "chunk/internal/proxy_base_inline.h" // the template bodies
