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

#pragma once

#include <type_traits> // std::conditional_t, std::is_trivially_copyable_v

namespace shrink_array
{

// ─────────────────────────────────────────────────────────────────────────────
// in_t<T> — small-type parameter helper
//
// Passes T by value when it is trivially copyable and ≤ 2*sizeof(void*)
// (16 bytes on 64-bit), else by const reference. For parameters of hot
// signatures that need not distinguish lvalue/rvalue arguments.
//
// ─── DECISION CHECKLIST FOR A NEW SIGNATURE ──────────────────────────────────
//
// For a parameter of type X, stop at the first "yes":
//
//   Q1. Is X deduced from the call-site argument (a function template
//       parameter)?  → `const X&` or `X&&`, NOT in_t<X>. Rule (1).
//   Q2. Does the body write, directly or transitively, to memory the caller
//       could alias through that parameter?  → `X v` by value, NOT in_t<X>.
//       Rule (2).
//   Q3. Otherwise → `in_t<X> v`: by value for small X, by const ref for large X.
//
// ─── RULE (1): DEDUCTION ─────────────────────────────────────────────────────
//
// `in_t<X>` is a non-deducible context (small_type<X> needs X first):
//
//     template <typename U>
//     void f(in_t<U> v);        // caller f(3.14) → "no matching function"
//
// Representative sites templated on U that keep `const U&` or a plain `U`
// (illustrative; grep `template <typename U>` in the Proxy hierarchy):
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
// -- DO take `in_t<U>`: their callers name U explicitly (`assign_impl<U>(v)`).
//
// ─── RULE (2): ALIASING ──────────────────────────────────────────────────────
//
// Do not use in_t<T> for a parameter when the body reads it and later
// overwrites memory it may alias: for large T, `in_t<T> = const T&` may bind to
// that memory and a later read returns the new value. Aggregate elements (a
// struct of three doubles is 24 bytes) make in_t<T> a `const T&` in practice,
// so the sites below keep `T v` for a real reason.
//
// The proxies' operator= (deduced, `const U&`) hand the operand to
// assign_impl<U>, whose in_t<U> is safe: the operand comes from the caller (a
// const read returns a VALUE, never a reference into a chunk buffer).
//
// Canonical alias site — writes the chunk's second-value slot through
// `ptr_second_value` (a SecondValueRef) after reading `second_v`, so a
// `const T&` bound to that slot would change under it:
//
//     TabCstVal2Holder::find_or_set_ind_cst_val2      (tab_cst_val2_holder.h)
//
// Also kept as `T v` by policy: no aliasing today, but on write paths where a
// large user T could introduce one:
//
//     TabCstVal2Holder::alloc_chunk(T val)            (tab_cst_val2_holder.h)
//     ArrayChunk2::set_second_value(T)                (array_chunk2.h and
//                                                       array_chunk2_inline.h)
//     Chunk2FastCompPolicy::have_chunk_to_compress(T) (chunk2_std.h)
//
// ─── COROLLARY FOR LOCAL VARIABLES ───────────────────────────────────────────
//
//     const T v_old = values[num];       // snapshot
//     ...                                // code that mutates values[num]
//     if (v_old == something) { ... }    // read snapshot
//
// The copy is INTENTIONAL. Do NOT turn these into `const T&`: that
// reintroduces the aliasing hazard. `const T v_old`, `v_new`, `current`,
// `second_v` in the Proxy inline files and chunk*_std_inline.h are all
// intentional snapshots.
//
// ─── ANTI-PATTERN: `const T x = expr; return x;` ─────────────────────────────
//
// Unrelated to in_t but often seen next to snapshots: `const` on a returned
// local kills NRVO (a const object cannot be moved), forcing a copy. Prefer
// `return expr;` (C++17 guaranteed elision).
//
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
concept small_type = std::is_trivially_copyable_v<T> && sizeof(T) <= 2 * sizeof(void*); // 16 bytes on 64-bit

template <typename T>
using in_t = std::conditional_t<small_type<T>, T, const T&>;

} // namespace shrink_array