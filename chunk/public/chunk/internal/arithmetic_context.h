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

#include <cstdint> // std::uint64_t (the liveness sentinel)

#include "chunk/internal/array_chunk_for.h"          // ArrayKey (ptr_on_array)
#include "chunk/internal/in_t.h"                     // InT<T>: the operands
#include "chunk/internal/deferred_lifecycle_check.h" // DEFERRED_LIFECYCLE sentinels

namespace shrink_array
{

namespace zero
{

template <typename T>
class Chunk;
template <typename T>
class Proxy;

} // namespace zero

/// @brief The proxy's num_tab: its armed value where the proxy keeps one
///        (get_num_tab_cached, zero), get_num_tab(array_ptr) where the proxy
///        takes the array, get_num_tab() otherwise.
template <typename ProxyT, typename Key>
constexpr int get_num_tab_helper(const ProxyT& proxy, Key array_ptr) noexcept
{
    if constexpr (requires { proxy.get_num_tab_cached(array_ptr); })
    {
        return proxy.get_num_tab_cached(array_ptr);
    }
    else if constexpr (requires { proxy.get_num_tab(array_ptr); })
    {
        return proxy.get_num_tab(array_ptr);
    }
    else
    {
        return proxy.get_num_tab();
    }
}

/**
 * @brief The minimal, safe context of a deferred write.
 * @details It stores ONLY the location of the write (chunk, index, logical
 *          array), the geometry frozen at enqueue, and, in the derived
 *          contexts, the operand / operation. It does NOT store the values
 *          pointer (reloaded from the chunk at replay), second_value / ind /
 *          inf_array (recomputed at replay), a Proxy instance (rebuilt if
 *          needed) or any cached compression metadata: a deferred write never
 *          uses stale state.
 */
template <typename T, typename ProxyT, typename ChunkT = zero::Chunk<T>>
struct BaseArithmeticContext
{
    /// @brief Capture the proxy's position, chunk, geometry and allocation key.
    /// @param prox Proxy the context is built from.
    /// @param array_ptr Owning array, stored in ptr_on_array and used to compute num_tab.
    BaseArithmeticContext(const ProxyT& prox, ArrayKey<ChunkT> array_ptr)
        : num(prox.get_num())
        , chunk(prox.get_chunk())
        , num_chunk(prox.get_num_chunk())
        , num_tab(get_num_tab_helper(prox, array_ptr))
        , ptr_on_array(array_ptr)
        , chunk_size_frozen(prox.chunk_size()) // resolve while the array is alive
        , alloc_key_frozen(prox.alloc_key())
    {
    }

    /// @brief Poison the liveness sentinel (magic = ctx_dead_magic).
    ~BaseArithmeticContext()
    {
        magic = deferred_lifecycle::ctx_dead_magic;
    }

    // Built in place in a pool block and owned by one deferred op: a copy
    // would be a second live context for the same write.
    BaseArithmeticContext(const BaseArithmeticContext&)            = delete;
    BaseArithmeticContext& operator=(const BaseArithmeticContext&) = delete;
    BaseArithmeticContext(BaseArithmeticContext&&)                 = delete;
    BaseArithmeticContext& operator=(BaseArithmeticContext&&)      = delete;

    int              num;          ///< index of the element in its chunk
    ChunkT&          chunk;        ///< the chunk written
    int              num_chunk;    ///< index of the chunk in its array
    int              num_tab;      ///< logical array index
    ArrayKey<ChunkT> ptr_on_array; ///< the owning array

    /// @brief Chunk geometry resolved AT ENQUEUE, while the source array is
    ///        guaranteed alive.
    /// @details Do not call chunk_size() at replay: under DynamicChunkSize it
    ///          dereferences ptr_on_array, and a deferred op can be replayed
    ///          (erase_own_rank_op / flush -> write_body) after that array is
    ///          destroyed (use-after-free; hidden by fast-alloc's immortal arenas
    ///          and by static builds). Geometry is immutable after construction,
    ///          so the frozen value is what a live re-read would return.
    unsigned int chunk_size_frozen = 0;
    /// @brief The array's slab-affinity key, frozen with the geometry and for the
    ///        same reason; replayed into the thread's allocation key.
    unsigned int alloc_key_frozen = 0;

    /// @brief DEFERRED_LIFECYCLE liveness sentinel, verified at the top of the
    ///        deferred write_body() before this ctx is dereferenced.
    /// @details Unconditional (not under a macro) so the layout is identical in
    ///          every TU whatever DEBUG/CHECK_OPTIM (no ODR skew); only the check in
    ///          DeferredOperation::write_body is gated. Poisoned in the destructor.
    std::uint64_t magic = deferred_lifecycle::ctx_alive_magic;
};

/// @brief Context of a deferred Proxy<T> write without operand or operation.
template <typename T>
struct ArithmeticProxyContext : public BaseArithmeticContext<T, zero::Proxy<T>>
{
    using Base = BaseArithmeticContext<T, zero::Proxy<T>>;

    /// @brief Build the context from @p proxy.
    /// @param proxy Proxy the context is built from.
    /// @param array_ptr Owning array.
    ArithmeticProxyContext(const zero::Proxy<T>& proxy, ArrayKey<zero::Chunk<T>> array_ptr) : Base(proxy, array_ptr) {}
};

/// @brief Context of a deferred Proxy<T> assignment: the assigned value.
template <typename T, typename U>
struct ArithmeticAffectationProxyContext : public BaseArithmeticContext<T, zero::Proxy<T>>
{
    using Base = BaseArithmeticContext<T, zero::Proxy<T>>;

    /// @brief Build the context from @p proxy with the value to assign.
    /// @param proxy Proxy the context is built from.
    /// @param op_value Assigned value, stored in operand.
    /// @param array_ptr Owning array.
    ArithmeticAffectationProxyContext(const zero::Proxy<T>& proxy, InT<U> op_value, ArrayKey<zero::Chunk<T>> array_ptr)
        : Base(proxy, array_ptr), operand(op_value)
    {
    }

    U operand; ///< assigned value
};

/// @brief Context of a deferred Proxy<T> compound operation (+=, -=, *= ...):
///        the operand and the operation.
template <typename T, typename Op>
struct ArithmeticOperationProxyContext : public BaseArithmeticContext<T, zero::Proxy<T>>
{
    using Base = BaseArithmeticContext<T, zero::Proxy<T>>;

    /// @brief Build the context from @p proxy with an operand and the operation to apply.
    /// @param proxy Proxy the context is built from.
    /// @param op_value Right-hand operand.
    /// @param op Functor implementing the operation.
    /// @param array_ptr Owning array.
    ArithmeticOperationProxyContext(const zero::Proxy<T>&    proxy,
                                    InT<T>                   op_value,
                                    Op                       op,
                                    ArrayKey<zero::Chunk<T>> array_ptr)
        : Base(proxy, array_ptr), operand(op_value), operation(op)
    {
    }

    T  operand;   ///< right-hand operand
    Op operation; ///< functor implementing the operation
};

/// @brief Context of a deferred zero_one_value::Proxy / zero_any::Proxy write without operation.
template <typename T, typename ChunkT, typename ProxyT>
struct ArithmeticContext : public BaseArithmeticContext<T, ProxyT, ChunkT>
{
    using Base = BaseArithmeticContext<T, ProxyT, ChunkT>;

    /// @brief Build the context from @p proxy.
    /// @param proxy Proxy the context is built from.
    /// @param array_ptr Owning array.
    ArithmeticContext(const ProxyT& proxy, ArrayKey<ChunkT> array_ptr) : Base(proxy, array_ptr) {}
};

/// @brief Context of a deferred zero_one_value::Proxy / zero_any::Proxy compound operation: the
///        operand and the operation.
template <typename T, typename Op, typename ChunkT, typename ProxyT>
struct ArithmeticOperationContext : public BaseArithmeticContext<T, ProxyT, ChunkT>
{
    using Base = BaseArithmeticContext<T, ProxyT, ChunkT>;

    /// @brief Build the context from @p proxy with an operand and the operation to apply.
    /// @param proxy Proxy the context is built from.
    /// @param op_value Right-hand operand.
    /// @param op Functor implementing the operation.
    /// @param array_ptr Owning array.
    ArithmeticOperationContext(const ProxyT& proxy, InT<T> op_value, Op op, ArrayKey<ChunkT> array_ptr)
        : Base(proxy, array_ptr), operand(op_value), operation(op)
    {
    }

    T  operand;   ///< right-hand operand
    Op operation; ///< functor implementing the operation
};

/// @brief Context of a deferred zero_one_value::Proxy / zero_any::Proxy assignment: the assigned value.
template <typename T, typename U, typename ChunkT, typename ProxyT>
struct ArithmeticAffectationContext : public BaseArithmeticContext<T, ProxyT, ChunkT>
{
    using Base = BaseArithmeticContext<T, ProxyT, ChunkT>;

    /// @brief Build the context from @p proxy with the value to assign.
    /// @param proxy Proxy the context is built from.
    /// @param op_value Assigned value, stored in operand.
    /// @param array_ptr Owning array.
    ArithmeticAffectationContext(const ProxyT& proxy, InT<U> op_value, ArrayKey<ChunkT> array_ptr)
        : Base(proxy, array_ptr), operand(op_value)
    {
    }

    U operand; ///< assigned value
};

} // namespace shrink_array
