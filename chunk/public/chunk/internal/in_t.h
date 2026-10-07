
#pragma once

/*
 * Copyright 2026 CEA (www.cea.fr)
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

#include <type_traits> // std::conditional_t, std::is_trivially_copyable_v

namespace shrink_array
{

// ─────────────────────────────────────────────────────────────────────────────
// in_t<T> — small-type parameter helper
//
// Passes T by value when it's trivially copyable and ≤ 2*sizeof(void*)
// (≤ 16 bytes on 64-bit), else by const reference. Intended for parameters of
// hot signatures that don't need to distinguish lvalue/rvalue arguments.
//
// ─── DECISION CHECKLIST FOR A NEW SIGNATURE ──────────────────────────────────
//
// When adding a function/method with a parameter of some type X, walk the
// questions in order. Stop at the first "yes":
//
//   Q1. Is X deduced from the argument at the call site (i.e. X is a
//       function template parameter, not fixed at the enclosing scope)?
//         → YES: use `const X&` or `X&&` (forwarding-ref). NOT in_t<X>.
//                See rule (1) below.
//
//   Q2. Does the function body write, directly or transitively, to memory
//       that the caller could alias via that parameter (e.g. writes through
//       a pointer whose target might be the same object as the argument)?
//         → YES: keep `X v` by value. NOT in_t<X>.
//                See rule (2) below.
//
//   Q3. Otherwise → use `in_t<X> v`. Optimal in both directions: by-value
//                   for small X (no indirection), by-const-ref for large X
//                   (no copy).
//
// ─── RULE (1): DEDUCTION ─────────────────────────────────────────────────────
//
// `in_t<X>` is a non-deducible context because computing `small_type<X>`
// requires X to be known first — chicken-and-egg. Example that FAILS:
//
//     template <typename U>
//     void f(in_t<U> v);        // caller f(3.14) → "no matching function"
//
// Keep `const U&` (or `U&&` forwarding-ref) in that case. Representative
// sites templated on U that keep `const U&` or a plain `U` (list is
// illustrative, not exhaustive — grep for `template <typename U>` in the
// Proxy hierarchy to find the current full set):
//
//     Proxy::operator=/+=/-=/*=//=/%=                 (proxy.h)
//     Proxy2::operator=                               (proxy2.h)
//     Proxy2A::operator=                              (proxy2A.h)
//     Proxy2Base::operator*=///=/%=                   (proxy2_base.h)
//     ProxyBase::operator+=/-=/&=/|=/^=/<<=/>>=       (proxy_base.h)
//     ProxyBase::enqueue_async_arithmetic             (proxy_base.h)
//     ProxyBase::apply_atomic_op (U parameter)        (proxy_base.h)
//     ProxyOperators::friend +,-,*,/                  (proxy_operators.h)
//     increm_chunk_write<V>(V v)                      (chunk_stats.h)
//     Proxy::is_cst_val_and_v_equal_zero              (proxy.h)
//
// The engines those operators call -- assign_impl (the three proxies),
// ProxyBase::arithmetic_assign / bitwise_assign, Proxy::apply_compound_operation
// -- DO take `in_t<U>`: their callers name U explicitly (`assign_impl<U>(v)`),
// so nothing is deduced there.
//
// ─── RULE (2): ALIASING ──────────────────────────────────────────────────────
//
// Do not use in_t<T> for a parameter that the function body both READS and
// later OVERWRITES some memory it may alias. For large T, `in_t<T> = const T&`
// may bind to memory that the function itself mutates, causing a subsequent
// read of the reference to return a stale value. Since the element contract
// opened to aggregates (a struct of three doubles is 24 bytes), in_t<T> IS
// `const T&` for some T in use: the sites below keep `T v` by value for that
// reason, not for a hypothetical future.
//
// The proxies' operator= are deduced sites too: they take `const U&` (one
// overload for a clean conversion, one deprecated for a risky one) and hand
// the operand to assign_impl<U>, whose in_t<U> copies a small operand once
// and passes a large one (a struct) by reference into the chunk write. No
// aliasing there: the operand comes from the caller (a const read returns a
// VALUE, never a reference into a chunk buffer).
//
// Canonical alias site — the function writes the chunk's second-value slot
// through `ptr_second_value` (a SecondValueRef) after reading `second_v`, so
// a `const T&` bound to that slot would change under it:
//
//     TabCstVal2Holder::find_or_set_ind_cst_val2      (tab_cst_val2_holder.h)
//
// Additional sites deliberately kept as `T v` under the same policy — no
// aliasing manifests today, but they participate in write paths where a
// future user extension with large T could introduce one:
//
//     TabCstVal2Holder::alloc_chunk(T val)            (tab_cst_val2_holder.h)
//     ArrayChunk2::set_second_value(T)                (array_chunk2.h and
//                                                       array_chunk2_inline.h)
//     Chunk2FastCompPolicy::have_chunk_to_compress(T) (chunk2_std.h)
//
// ─── COROLLARY FOR LOCAL VARIABLES ───────────────────────────────────────────
//
// The aliasing rule extends to local snapshots of the form
//
//     const T v_old = values[num];       // snapshot
//     ...                                // code that mutates values[num]
//     if (v_old == something) { ... }    // read snapshot
//
// The copy is INTENTIONAL to isolate the snapshot from the subsequent write.
// Do NOT "optimise" these to `const T&` — that would reintroduce the aliasing
// hazard the outer function was carefully avoiding. `const T v_old`,
// `const T v_new`, `const T current`, `const T second_v` throughout the
// Proxy inline files and chunk*_std_inline.h are all intentional snapshots.
//
// ─── ANTI-PATTERN: `const T x = expr; return x;` (unrelated but tempting) ───
//
// Not covered by in_t<T> at all, but often noticed at the same time as the
// snapshot pattern above. The `const` on a local kills NRVO on the return
// (the compiler cannot move a const object), forcing a copy at the return
// site for any T that isn't cheaply copyable. Prefer a direct `return expr;`
// — C++17 mandatory copy elision applies to the returned prvalue and
// removes the copy entirely.
//
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
concept small_type = std::is_trivially_copyable_v<T> && sizeof(T) <= 2 * sizeof(void*); // 16 bytes on 64-bit

template <typename T>
using in_t = std::conditional_t<small_type<T>, T, const T&>;

} // namespace shrink_array