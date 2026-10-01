#pragma once
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

#include <cstdint> // std::uint64_t (the generation counter)
#include <tuple>

#include "chunk/internal/in_t.h"                     // in_t<T>: the operands
#include "chunk/internal/deferred_lifecycle_check.h" // DEFERRED_LIFECYCLE sentinels

namespace shrink_array
{

template <typename T>
class Chunk;

template <typename T>
class Proxy;

// ============================================================================
// Helper: dispatch get_num_tab() depending on Proxy signature
// ============================================================================
template <typename ProxyType>
constexpr int get_num_tab_helper(const ProxyType& proxy, const void* array_ptr) noexcept
{
    if constexpr (requires { proxy.get_num_tab(array_ptr); })
        return proxy.get_num_tab(array_ptr);
    else
        return proxy.get_num_tab();
}

// ============================================================================
// BaseArithmeticContext
//
// This is the *safe* minimal context for asynchronous operations.
// It stores ONLY:
//   - the location of the write (chunk, index, logical array)
//   - the operand/operation (in derived classes)
//
// It does NOT store:
//   - values pointer (must be reloaded from chunk at execution time)
//   - second_value / ind / inf_array (must be recomputed at execution time)
//   - a Proxy instance (must be reconstructed if needed)
//   - any cached compression metadata
//
// This guarantees that async operations never use stale state.
// ============================================================================
template <typename T, typename ProxyType, typename ChunkType = Chunk<T>>
struct BaseArithmeticContext
{
    // Only for proxies that expose get_inf_array()
    using inf_array_t = std::tuple<T, int, int>;

    /// @brief Capture the proxy's position, chunk, geometry and allocation key.
    /// @param prox Proxy the context is built from.
    /// @param array_ptr Owning array, stored in ptr_on_array and used to compute num_tab.
    BaseArithmeticContext(const ProxyType& prox, const void* array_ptr)
        : num(prox.get_num())
        , chunk(prox.get_chunk())
        , num_chunk(prox.get_num_chunk())
        , num_tab(get_num_tab_helper(prox, array_ptr))
        , ptr_on_array(array_ptr)
        , chunk_size_frozen(prox.chunk_size()) // resolve while the array is alive
        , alloc_key_frozen(prox.alloc_key())
    {
        if constexpr (requires { prox.get_inf_array(); })
        {
            inf_array_ptr = &prox.get_inf_array();
        }
    }

    /// @brief Poison the liveness sentinel (magic = ctx_dead_magic).
    ~BaseArithmeticContext()
    {
        magic = deferred_lifecycle::ctx_dead_magic;
    }

    int         num;
    ChunkType&  chunk;
    int         num_chunk;
    int         num_tab;
    const void* ptr_on_array;

    // fw14 (dyn UAF fix): chunk geometry resolved AT ENQUEUE, while the source
    // array is guaranteed alive. Under DynamicChunkSize the proxy's chunk_size()
    // dereferences ptr_on_array (the ArrayChunk); but a deferred op can be
    // replayed (erase_own_rank_op / flush -> write_body) AFTER that array has
    // been destroyed, so the late chunk_size() was a use-after-free — a SIGSEGV
    // under std-alloc (page actually freed) that fast-alloc masked (immortal
    // arenas keep the page mapped) and static builds masked (size folds to a
    // compile-time constant, ptr not really loaded). Freezing the value here and
    // reading it back on the replay path removes the late dereference entirely.
    // Geometry is immutable after array construction, so the frozen value is
    // exactly what a live re-read would have returned.
    unsigned int chunk_size_frozen = 0;
    /// @brief The array's slab-affinity key, frozen with the geometry and for the
    ///        same reason; replayed into the thread's allocation key.
    unsigned int alloc_key_frozen = 0;

  private:
    const inf_array_t* inf_array_ptr = nullptr;

  public:
    // DEFERRED_LIFECYCLE: liveness sentinel, verified at the top of
    // the deferred write_body() before this ctx is dereferenced. Kept
    // UNCONDITIONAL (not under a macro) so the struct layout is identical across
    // every TU regardless of DEBUG/CHECK_OPTIM — no ODR/layout skew. Only the
    // *checking* is gated (in DeferredOperation::write_body). Poisoned in the
    // destructor so a use-after-recycle read no longer sees ctx_alive_magic.
    std::uint64_t magic = deferred_lifecycle::ctx_alive_magic;
};

// ============================================================================
// ArithmeticProxyContext
// For Proxy<T> without operand or operation.
// ============================================================================

template <typename T>
struct ArithmeticProxyContext : BaseArithmeticContext<T, Proxy<T>>
{
    using Base = BaseArithmeticContext<T, Proxy<T>>;

    /// @brief Build the context from @p proxy.
    /// @param proxy Proxy the context is built from.
    /// @param array_ptr Owning array.
    ArithmeticProxyContext(const Proxy<T>& proxy, const void* array_ptr) : Base(proxy, array_ptr) {}
};

// ============================================================================
// ArithmeticAffectationProxyContext
// For Proxy<T> with a simple assignment operand.
// ============================================================================

template <typename T, typename U>
struct ArithmeticAffectationProxyContext : BaseArithmeticContext<T, Proxy<T>>
{
    using Base = BaseArithmeticContext<T, Proxy<T>>;

    /// @brief Build the context from @p proxy with the value to assign.
    /// @param proxy Proxy the context is built from.
    /// @param op_value Assigned value, stored in operand.
    /// @param array_ptr Owning array.
    ArithmeticAffectationProxyContext(const Proxy<T>& proxy, in_t<U> op_value, const void* array_ptr)
        : Base(proxy, array_ptr), operand(op_value)
    {
    }

    U operand; // Assigned value
};

// ============================================================================
// ArithmeticOperationProxyContext
// For Proxy<T> with operand + operation (e.g. +=, -=, *=).
// ============================================================================

template <typename T, typename Op>
struct ArithmeticOperationProxyContext : BaseArithmeticContext<T, Proxy<T>>
{
    using Base = BaseArithmeticContext<T, Proxy<T>>;

    /// @brief Build the context from @p proxy with an operand and the operation to apply.
    /// @param proxy Proxy the context is built from.
    /// @param op_value Right-hand operand.
    /// @param op Functor implementing the operation.
    /// @param array_ptr Owning array.
    ArithmeticOperationProxyContext(const Proxy<T>& proxy, in_t<T> op_value, Op op, const void* array_ptr)
        : Base(proxy, array_ptr), operand(op_value), operation(op)
    {
    }

    T  operand;   // Right-hand operand
    Op operation; // Functor implementing the operation
};

// ============================================================================
// ArithmeticContext
// Generic context for Proxy2 / Proxy2A without operation.
// ============================================================================

template <typename T, typename ChunkType, typename ProxyType>
struct ArithmeticContext : BaseArithmeticContext<T, ProxyType, ChunkType>
{
  public:
    using Base = BaseArithmeticContext<T, ProxyType, ChunkType>;

    /// @brief Build the context from @p proxy.
    /// @param proxy Proxy the context is built from.
    /// @param array_ptr Owning array.
    ArithmeticContext(const ProxyType& proxy, const void* array_ptr) : Base(proxy, array_ptr) {}
};

// ============================================================================
// ArithmeticOperationContext
// Generic context for Proxy2 / Proxy2A with operand + operation.
// ============================================================================
template <typename T, typename Op, typename ChunkType, typename ProxyType>
struct ArithmeticOperationContext : BaseArithmeticContext<T, ProxyType, ChunkType>
{
    using Base = BaseArithmeticContext<T, ProxyType, ChunkType>;

    /// @brief Build the context from @p proxy with an operand and the operation to apply.
    /// @param proxy Proxy the context is built from.
    /// @param op_value Right-hand operand.
    /// @param op Functor implementing the operation.
    /// @param array_ptr Owning array.
    ArithmeticOperationContext(const ProxyType& proxy, in_t<T> op_value, Op op, const void* array_ptr)
        : Base(proxy, array_ptr), operand(op_value), operation(op)
    {
    }

    T  operand;
    Op operation;
};

// ============================================================================
// ArithmeticAffectationContext
// Generic assignment context for Proxy2 / Proxy2A.
// ============================================================================

template <typename T, typename U, typename ChunkType, typename ProxyType>
struct ArithmeticAffectationContext : BaseArithmeticContext<T, ProxyType, ChunkType>
{
    using Base = BaseArithmeticContext<T, ProxyType, ChunkType>;

    /// @brief Build the context from @p proxy with the value to assign.
    /// @param proxy Proxy the context is built from.
    /// @param op_value Assigned value, stored in operand.
    /// @param array_ptr Owning array.
    ArithmeticAffectationContext(const ProxyType& proxy, in_t<U> op_value, const void* array_ptr)
        : Base(proxy, array_ptr), operand(op_value)
    {
    }

    U operand;
};

} // namespace shrink_array
