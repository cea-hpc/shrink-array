/*
 * Copyright 2025-2026 CEA (www.cea.fr)
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

#include <cstdint> // uint8_t (EventCode underlying type)
#include <cstdio>  // std::fprintf/std::snprintf — used HERE, not via includers

#include "chunk/config.h"                   // chunk_config::ring_log_debug, CHUNK_RING_LOG_ANY
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED (Manual mode: end() not called)

namespace shrink_array
{

// ----------------------------- Simple ring log for debugging (lock-free-ish) -----------------------------
enum EventCode : uint8_t
{
    ev_begin_slow_path = 1,
    ev_end_slow_path,
    ev_begin_fast_path,
    ev_end_fast_path,
    ev_compress_attempt,
    ev_compress_started,
    ev_compress_aborted,
    ev_compress_finished,
    // ─── cst_val transitions ────────────────────────────────────────
    ev_cas_values_to_cst,   // info = chunk_idx, sub-value = which cst (1 or 2)
    ev_cas_values_from_cst, // info = chunk_idx (create_value success)
    ev_cas_values_fail,     // info = chunk_idx (create_value failed = recup)
    ev_flush_begin,         // info = chunk_idx
    ev_flush_end,           // info = chunk_idx
    ev_lambda_async_assign, // info = chunk_idx (in lambda exec)
    ev_lambda_async_arith,  // info = chunk_idx
    ev_values_load_is_cst,  // info = chunk_idx (RACE DETECTOR if observed in lambda)
    ev_store_value,         // info = (chunk_idx << 8) | num
    ev_policy_begin_write,  // info = thr_rank (entered begin_write)
    ev_policy_end_write,    // info = thr_rank (entered end_write)
    // ─── thread-ops bitmap tracking ──────────────────────────────────
    ev_mark_thread_has_ops,  // info = thr_rank
    ev_clear_thread_has_ops, // info = thr_rank
    // ─── op lifecycle (use-after-free via op_map) ────────────────────
    ev_op_alloc,     // info = num ; track op storage alloc
    ev_op_enqueue,   // info = (chunk_idx << 8) | num ; put in op_map
    ev_op_free,      // info = num ; op block recycled (free_op)
    ev_op_flush_one, // info = num ; about to write_body this op
    // ─── chunk VALUE storage (op/value address aliasing) ─────────────
    ev_value_alloc, // a chunk values buffer was allocated at addr
    ev_value_free,  // a chunk values buffer was freed at addr
};

#ifdef CHUNK_RING_LOG_ANY

/// @brief One ring-buffer record: timestamp, thread id, event, payload and (op events) address.
struct RingEntry
{
    uint64_t  ts;
    uint32_t  tid;
    EventCode ev;
    uint32_t  info;
    uint64_t  addr; // op address for EV_OP_* events (0 otherwise)
};

/// @brief Record a begin event, timestamped with the serialising start counter read.
/// @param tid Thread id (see get_tid()).
/// @param ev Event code.
/// @param info Payload.
extern void ring_log_start(uint32_t tid, EventCode ev, uint32_t info = 0);

/// @brief Record an end event, timestamped with the serialising end counter read.
/// @param tid Thread id (see get_tid()).
/// @param ev Event code.
/// @param info Payload.
extern void ring_log_end(uint32_t tid, EventCode ev, uint32_t info = 0);

/// @brief Record an event in the ring buffer.
/// @param tid Thread id (see get_tid()).
/// @param ev Event code.
/// @param info Payload.
extern void ring_log(uint32_t tid, EventCode ev, uint32_t info = 0);

// Log an op-lifecycle event with its full address (48-bit), to correlate
// alloc/enqueue/free/flush of the SAME op block across threads.
extern void ring_log_op(uint32_t tid, EventCode ev, uint32_t num, const void* addr);

/// @brief Id recorded for the calling thread: its OpenMP team rank, or a 32-bit hash of std::thread::id
///        without OpenMP.
extern uint32_t get_tid();

/// @brief Print every entry still held in the ring to std::cout.
extern void dump_ring();

/// @brief Print the entries whose address is @p target, then those in the same 64 KiB block.
/// @param target Address to look for.
extern void dump_ring_for_addr(const void* target);

/// @brief Reset the ring head to 0.
extern void ring_init();

/// @brief Find overlaps between fast-path and compression intervals in the ring and report them on std::cout.
/// @return true when a thread's fast-path interval overlaps a compression interval of the same thread.
extern bool analyze_ring_sweep();

#else

/// @brief Ring log disabled: always 0.
inline uint32_t get_tid()
{
    return 0;
}

/// @brief Ring log disabled: no-op.
inline void ring_log(uint32_t /*tid*/, EventCode /*ev*/, uint32_t /*info*/ = 0) {}

/// @brief Ring log disabled: no-op.
inline void ring_log_end(uint32_t /*tid*/, EventCode /*ev*/, uint32_t /*info*/ = 0) {}

/// @brief Ring log disabled: no-op.
inline void ring_log_start(uint32_t /*tid*/, EventCode /*ev*/, uint32_t /*info*/ = 0) {}

/// @brief Ring log disabled: no-op.
inline void ring_log_op(uint32_t /*tid*/, EventCode /*ev*/, uint32_t /*num*/, const void* /*addr*/) {}

/// @brief Ring log disabled: no-op.
inline void dump_ring() {}

/// @brief Ring log disabled: no-op.
inline void dump_ring_for_addr(const void* /*target*/) {}

// No-op counterparts too, so `if constexpr (chunk_config::ring_log_debug)`
// compiles and links in every build without an #ifdef at the call site.
inline void ring_init() {}

/// @brief Ring log disabled: always false.
inline bool analyze_ring_sweep()
{
    return false;
}

#endif
/// @brief RAII helper for ring-log paired begin/end events.
///
/// On construction: emits BeginEvent with the (optional) payload.
/// On destruction:  emits EndEvent with the same payload.
///
/// `Manual` mode disables the destructor-emitted end. Caller calls .end()
/// explicitly. Useful when the end must happen at a precise point (after a
/// CAS, before a return) rather than at scope exit.
template <EventCode BeginEvent, EventCode EndEvent, bool Manual = false>
class ScopedRingLog
{
  public:
    /// @param payload_value Optional payload. 0 means no payload.
    explicit ScopedRingLog(uint32_t payload_value = 0) noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            tid_     = get_tid();
            payload_ = payload_value;
            ended_   = false;
            ring_log_start(tid_, BeginEvent, payload_);
        }
    }

    /// @brief Emit EndEvent unless in Manual mode; in Manual mode, check builds print a message on stderr when
    ///        end() was not called.
    ~ScopedRingLog() noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            if constexpr (!Manual)
            {
                ring_log_end(tid_, EndEvent, payload_); // end, not start
            }
            else
            {
                // Manual mode: stop (in check builds only) if the user forgot to call .end()
                if constexpr (build_config::any_check)
                {
                    if (!ended_)
                    {
                        CHUNK_INVARIANT_FAILED("ScopedRingLog<Manual>: end() not called before destruction "
                                               "(BEGIN_EVENT={})",
                                               static_cast<int>(BeginEvent));
                    }
                }
            }
        }
    }

    // Non-copyable, non-movable: a scoped logger should not migrate scopes
    ScopedRingLog(const ScopedRingLog&)            = delete;
    ScopedRingLog& operator=(const ScopedRingLog&) = delete;
    ScopedRingLog(ScopedRingLog&&)                 = delete;
    ScopedRingLog& operator=(ScopedRingLog&&)      = delete;

    /// Explicit end. For Manual mode. Idempotent: calling twice is OK.
    void end() noexcept
    {
        if constexpr (chunk_config::ring_log_debug && Manual)
        {
            if (!ended_)
            {
                ring_log_end(tid_, EndEvent, payload_);
                ended_ = true;
            }
        }
    }

  private:
    [[maybe_unused]] uint32_t tid_;
    [[maybe_unused]] uint32_t payload_;
    [[maybe_unused]] bool     ended_;
};

// Convenience aliases
template <EventCode BeginEvent, EventCode EndEvent>
using ScopedRingLogManualEnd = ScopedRingLog<BeginEvent, EndEvent, /*Manual=*/true>;


} // namespace shrink_array