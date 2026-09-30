//
// Created by ballereaup on 13/10/2025.
//
#pragma once

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

#include <cstdint> // uint8_t (EventCode underlying type)
#include <cstdio>  // std::fprintf/std::snprintf — used HERE, not via includers

#include "chunk/config.h"

namespace shrink_array
{

//#define RING_LOG_DEBUG

// ----------------------------- Simple ring log for debugging (lock-free-ish) -----------------------------
enum EventCode : uint8_t
{
    EV_BEGIN_SLOW_PATH = 1,
    EV_END_SLOW_PATH,
    EV_BEGIN_FAST_PATH,
    EV_END_FAST_PATH,
    EV_COMPRESS_ATTEMPT,
    EV_COMPRESS_STARTED,
    EV_COMPRESS_ABORTED,
    EV_COMPRESS_FINISHED,
    // ─── new events for cst_val corruption tracking ─────────────────
    EV_CAS_VALUES_TO_CST,    // info = chunk_idx, sub-value = which cst (1 or 2)
    EV_CAS_VALUES_FROM_CST,  // info = chunk_idx (create_value success)
    EV_CAS_VALUES_FAIL,      // info = chunk_idx (create_value failed = recup)
    EV_FLUSH_BEGIN,          // info = chunk_idx
    EV_FLUSH_END,            // info = chunk_idx
    EV_LAMBDA_ASYNC_ASSIGN,  // info = chunk_idx (in lambda exec)
    EV_LAMBDA_ASYNC_ARITH,   // info = chunk_idx
    EV_VALUES_LOAD_IS_CST,   // info = chunk_idx (RACE DETECTOR if observed in lambda)
    EV_STORE_VALUE,          // info = (chunk_idx << 8) | num
    EV_POLICY_BEGIN_WRITE,   // info = thr_rank (entered begin_write)
    EV_POLICY_END_WRITE,     // info = thr_rank (entered end_write) // ─── new events for thread-ops bitmap tracking
                             // ──────────────────
    EV_MARK_THREAD_HAS_OPS,  // info = thr_rank
    EV_CLEAR_THREAD_HAS_OPS, // info = thr_rank
    // ─── SEGV hunt: op lifecycle (use-after-free via op_map) ─────────
    EV_OP_ALLOC,     // info = num ; track op storage alloc
    EV_OP_ENQUEUE,   // info = (chunk_idx << 8) | num ; put in op_map
    EV_OP_FREE,      // info = num ; op block recycled (free_op)
    EV_OP_FLUSH_ONE, // info = num ; about to write_body this op
    // ─── SEGV hunt: chunk VALUE storage (to prove op/value address aliasing) ──
    EV_VALUE_ALLOC, // a chunk values buffer was allocated at addr
    EV_VALUE_FREE,  // a chunk values buffer was freed at addr
};

#ifdef RING_LOG_ANY

/// @brief One ring-buffer record: timestamp, thread id, event, payload and (op events) address.
struct RingEntry
{
    uint64_t  ts;
    uint32_t  tid;
    EventCode ev;
    uint32_t  info;
    uint64_t  addr; // SEGV hunt: op address for EV_OP_* events (0 otherwise)
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

// SEGV hunt: log an op-lifecycle event with its full address (48-bit) so we can
// correlate alloc/enqueue/free/flush of the SAME op block across threads.
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
inline void ring_log(uint32_t, EventCode, uint32_t = 0) {}

/// @brief Ring log disabled: no-op.
inline void ring_log_end(uint32_t, EventCode, uint32_t = 0) {}

/// @brief Ring log disabled: no-op.
inline void ring_log_start(uint32_t, EventCode, uint32_t = 0) {}

/// @brief Ring log disabled: no-op.
inline void ring_log_op(uint32_t, EventCode, uint32_t, const void*) {}

/// @brief Ring log disabled: no-op.
inline void dump_ring() {}

/// @brief Ring log disabled: no-op.
inline void dump_ring_for_addr(const void*) {}

// These two used to have no no-op counterpart, which made them the only names of
// this header a caller could not spell without an #ifdef of its own. With the
// stubs, `if constexpr (chunk_config::ring_log_debug)` compiles and links in
// every build, and shrink_bench.cpp no longer needs the directive.
inline void ring_init() {}

/// @brief Ring log disabled: always false.
inline bool analyze_ring_sweep()
{
    return false;
}

#endif
/// @brief RAII helper for ring-log paired begin/end events.
///
/// On construction: emits BEGIN_EVENT with the (optional) payload.
/// On destruction:  emits END_EVENT with the same payload.
///
/// `Manual` mode disables the destructor-emitted end. Caller calls .end()
/// explicitly. Useful when the end must happen at a precise point (after a
/// CAS, before a return) rather than at scope exit.
template <EventCode BEGIN_EVENT, EventCode END_EVENT, bool Manual = false>
struct ScopedRingLog
{
    /// @param payload_value Optional payload. 0 means no payload.
    explicit ScopedRingLog(uint32_t payload_value = 0) noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            tid     = get_tid();
            payload = payload_value;
            ended   = false;
            ring_log_start(tid, BEGIN_EVENT, payload);
        }
    }

    /// @brief Emit END_EVENT unless in Manual mode; in Manual mode, check builds print a message on stderr when
    ///        end() was not called.
    ~ScopedRingLog() noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            if constexpr (!Manual)
            {
                ring_log_end(tid, END_EVENT, payload); // ← fix bug 1: end, not start
            }
            else
            {
                // Manual mode: warn (in debug only) if user forgot to call .end()
                if constexpr (build_config::any_check)
                {
                    if (!ended)
                    {
                        std::fprintf(stderr,
                                     "[INVARIANT] ScopedRingLog<Manual>: end() not called "
                                     "before destruction (BEGIN_EVENT=%d)\n",
                                     static_cast<int>(BEGIN_EVENT));
                    }
                }
            }
        }
    }

    /// Explicit end. For Manual mode. Idempotent: calling twice is OK.
    void end() noexcept
    {
        if constexpr (chunk_config::ring_log_debug && Manual)
        {
            if (!ended)
            {
                ring_log_end(tid, END_EVENT, payload);
                ended = true;
            }
        }
    }

    // Non-copyable, non-movable: a scoped logger should not migrate scopes
    ScopedRingLog(const ScopedRingLog&)            = delete;
    ScopedRingLog& operator=(const ScopedRingLog&) = delete;
    ScopedRingLog(ScopedRingLog&&)                 = delete;
    ScopedRingLog& operator=(ScopedRingLog&&)      = delete;

  private:
    [[maybe_unused]] uint32_t tid;
    [[maybe_unused]] uint32_t payload;
    [[maybe_unused]] bool     ended;
};

// Convenience aliases
template <EventCode BEGIN_EVENT, EventCode END_EVENT>
using ScopedRingLogManualEnd = ScopedRingLog<BEGIN_EVENT, END_EVENT, /*Manual=*/true>;


} // namespace shrink_array