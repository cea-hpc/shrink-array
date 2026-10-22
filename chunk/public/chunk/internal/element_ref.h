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
 * @file element_ref.h
 * @brief ElementRef: what the non-const operator[] (at, front, back) returns --
 *        the write proxy, usable as a temporary only.
 *
 * @details `auto v = a[i];` keeps the PROXY, where std::vector's `auto` copies
 *          the value: v follows the element and dies with the array's storage
 *          (resize, re-chunk, ...). Code ported from std::vector wants
 *          `T v = a[i];`. The declaration itself cannot be refused: since
 *          C++17 `auto v = a[i];` constructs v in place from the prvalue.
 *
 *          Every USE of v is refused instead: each operation of ElementRef is
 *          &&-qualified (a temporary: `a[i] = x`, `T y = a[i]`, `a[i] + b[j]`,
 *          ...), and its &-qualified twin -- selected by a NAMED ElementRef --
 *          is a static_assert with the message below. The library still names
 *          proxies internally (ArrayChunkBase::element_proxy). Every operation
 *          is force-inlined so the optimised code is the bare proxy's (GCC may
 *          otherwise keep e.g. a complex `*=` out of line).
 *          -DCHUNK_LOOSE_ELEMENT_REF returns the bare proxy
 *          (build_config::strict_element_ref).
 *
 *          The iterators yield values already (`for (auto x : a)` copies).
 *          A generic function that takes a[i] by value or by forwarding
 *          reference and then uses its parameter trips the same assertion:
 *          convert first (`f(T(a[i]))`), or forward the parameter
 *          (`std::forward<decltype(e)>(e) = x`, which std::vector code accepts
 *          too).
 *
 *          A proxy meant to be kept: `auto&& r = a[i]; std::move(r) = x;` --
 *          std::move says so, and the rules of the proxy apply (it must not
 *          outlive the storage).
 */

#include <ostream>     // std::basic_ostream (operator<<)
#include <type_traits> // std::remove_cvref_t, std::is_same_v
#include <utility>     // std::move, std::forward

#include "chunk/internal/def_force_inline.h" // CHUNK_FORCE_INLINE: the wrapper must cost nothing

namespace shrink_array
{

template <typename P, typename T>
class ElementRef;

namespace element_ref_detail
{
/// @brief Always false, but only once instantiated: the static_assert of the
///        &-qualified operators fires when one of them is selected, not before.
template <typename...>
inline constexpr bool misuse = false;

/// @brief Is @p X a specialisation of ElementRef?
template <typename X>
inline constexpr bool is_element_ref = false;
/// @brief A specialisation of ElementRef is one.
template <typename P, typename T>
inline constexpr bool is_element_ref<ElementRef<P, T>> = true;

/// @brief The operand an ElementRef hands to its proxy: the other side's proxy
///        when it is an ElementRef too (so proxy-to-proxy operations keep their
///        own semantics), the operand itself otherwise.
template <typename U>
CHUNK_FORCE_INLINE constexpr decltype(auto) unwrap(U&& u) noexcept
{
    if constexpr (is_element_ref<std::remove_cvref_t<U>>)
    {
        static_assert(!std::is_lvalue_reference_v<U>,
                      ""
                      "a named element proxy is used as an operand: `auto v = a[i];` keeps a PROXY that follows the "
                      "element -- write `T v = a[i];` (or pass std::move(v) where the proxy is meant)");
        return std::forward<U>(u).proxy();
    }
    else
    {
        return std::forward<U>(u);
    }
}

} // namespace element_ref_detail

/// @brief The message of every refused use of a named ElementRef.
#define CHUNK_ELEMENT_REF_MISUSE                                                                                 \
    "a named element proxy is used: `auto v = a[i];` keeps a PROXY that follows the element and dies with the "  \
    "array's storage -- write `T v = a[i];` (or std::move(v) where the proxy is meant; a generic function that " \
    "takes a[i] by value or forwarding reference: pass T(a[i]))"

/**
 * @brief The write proxy @p P of an element of type @p T, usable as a temporary
 *        only (see the file comment).
 * @tparam P the proxy type (Proxy<T>, Proxy2<T>, Proxy2A<T>)
 * @tparam T the element type
 */
template <typename P, typename T>
class ElementRef
{
  public:
    using ProxyType  = P; ///< The wrapped proxy.
    using value_type = T; ///< The element type.

    /// @brief Wrap @p p (operator[] and the operators below).
    CHUNK_FORCE_INLINE explicit ElementRef(P&& p) noexcept : p_(std::move(p)) {}

    /// @brief Moved (std::move(v), an operator's result).
    ElementRef(ElementRef&&) noexcept = default;
    /// @brief Refused: a copy is a named proxy's doing (`auto w = v;`).
    ElementRef(const ElementRef& other) : p_(other.p_)
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
    }
    ~ElementRef() = default;

    // ------------------------------------------------------------------ read
    /// @brief The element's value (the proxy's gated read).
    CHUNK_FORCE_INLINE operator T() const&& noexcept(noexcept(static_cast<T>(std::declval<const P&>())))
    {
        return static_cast<T>(p_);
    }
    /// @brief Refused: the conversion of a named element proxy.
    operator T() const&
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
        return static_cast<T>(p_);
    }

    /// @brief ArrayChunkBits: `bool b = a[i]`, the proxy's operator bool (a
    ///        Bit converts to bool, but C++ does not chain the two conversions).
    CHUNK_FORCE_INLINE operator bool() const&&
        requires(!std::is_same_v<T, bool>) && requires(const P& p) { p.operator bool(); }
    {
        return p_.operator bool();
    }
    /// @brief Refused: the bool conversion of a named element proxy.
    operator bool() const&
        requires(!std::is_same_v<T, bool>) && requires(const P& p) { p.operator bool(); }
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
        return false;
    }

    // ----------------------------------------------------------- assignment
    /// @brief `a[i] = b[j]`: the proxy-to-proxy assignment (gated read of b[j]).
    CHUNK_FORCE_INLINE decltype(auto) operator=(ElementRef&& from) && noexcept
    {
        return wrap(p_ = from.p_);
    }
    /// @brief Refused: assignment to a named element proxy.
    ElementRef& operator=(ElementRef&& /*from*/) & noexcept
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
        return *this;
    }
    /// @brief Refused: `a[i] = v`, v a named element proxy.
    ElementRef& operator=(const ElementRef& /*from*/) &&
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
        return *this;
    }
    /// @brief Refused: `w = v`, both named element proxies.
    ElementRef& operator=(const ElementRef& /*from*/) &
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
        return *this;
    }

    /// @brief `a[i] = x`: the proxy's assignment.
    template <typename U>
        requires(!std::is_same_v<std::remove_cvref_t<U>, ElementRef>) &&
                requires(P& p, U&& u) { p = element_ref_detail::unwrap(std::forward<U>(u)); }
    CHUNK_FORCE_INLINE decltype(auto) operator=(U&& u) &&
    {
        return wrap(p_ = element_ref_detail::unwrap(std::forward<U>(u)));
    }
    /// @brief Refused: assignment to a named element proxy.
    template <typename U>
        requires(!std::is_same_v<std::remove_cvref_t<U>, ElementRef>) &&
                requires(P& p, U&& u) { p = std::forward<U>(u); }
    ElementRef& operator=(U&& /*u*/) &
    {
        static_assert(element_ref_detail::misuse<U>, CHUNK_ELEMENT_REF_MISUSE);
        return *this;
    }

// The compound assignments: forwarded to the proxy where it has them.
// OP is an operator token (+=, ...), which cannot be parenthesized.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define CHUNK_ELEMENT_REF_COMPOUND(OP)                                                          \
    template <typename U>                                                                       \
        requires requires(P& p, U&& u) { p OP element_ref_detail::unwrap(std::forward<U>(u)); } \
    CHUNK_FORCE_INLINE decltype(auto) operator OP(U&& u) &&                                     \
    {                                                                                           \
        return wrap(p_ OP element_ref_detail::unwrap(std::forward<U>(u)));                      \
    }                                                                                           \
    template <typename U>                                                                       \
        requires requires(P& p, U&& u) { p OP std::forward<U>(u); }                             \
    ElementRef& operator OP(U&& /*u*/) &                                                        \
    {                                                                                           \
        static_assert(element_ref_detail::misuse<U>, CHUNK_ELEMENT_REF_MISUSE);                 \
        return *this;                                                                           \
    }
    // NOLINTEND(bugprone-macro-parentheses)

    CHUNK_ELEMENT_REF_COMPOUND(+=)
    CHUNK_ELEMENT_REF_COMPOUND(-=)
    CHUNK_ELEMENT_REF_COMPOUND(*=)
    CHUNK_ELEMENT_REF_COMPOUND(/=)
    CHUNK_ELEMENT_REF_COMPOUND(%=)
    CHUNK_ELEMENT_REF_COMPOUND(&=)
    CHUNK_ELEMENT_REF_COMPOUND(|=)
    CHUNK_ELEMENT_REF_COMPOUND(^=)
    CHUNK_ELEMENT_REF_COMPOUND(<<=)
    CHUNK_ELEMENT_REF_COMPOUND(>>=)
#undef CHUNK_ELEMENT_REF_COMPOUND

    // ------------------------------------------------- increment, decrement
    /// @brief `++a[i]`.
    CHUNK_FORCE_INLINE decltype(auto) operator++() &&
        requires requires(P& p) { ++p; }
    {
        return wrap(++p_);
    }
    /// @brief `--a[i]`.
    CHUNK_FORCE_INLINE decltype(auto) operator--() &&
        requires requires(P& p) { --p; }
    {
        return wrap(--p_);
    }
    /// @brief `a[i]++`.
    CHUNK_FORCE_INLINE decltype(auto) operator++(int) &&
        requires requires(P& p) { p++; }
    {
        return wrap(p_++);
    }
    /// @brief `a[i]--`.
    CHUNK_FORCE_INLINE decltype(auto) operator--(int) &&
        requires requires(P& p) { p--; }
    {
        return wrap(p_--);
    }
    /// @brief Refused: increment of a named element proxy.
    ElementRef& operator++() &
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
        return *this;
    }
    /// @brief Refused: decrement of a named element proxy.
    ElementRef& operator--() &
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
        return *this;
    }
    /// @brief Refused: increment of a named element proxy.
    ElementRef& operator++(int) &
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
        return *this;
    }
    /// @brief Refused: decrement of a named element proxy.
    ElementRef& operator--(int) &
    {
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE);
        return *this;
    }

    // --------------------------------------------------------- comparisons
    /// @brief `a[i] == x` (rewritten: `x == a[i]`, `a[i] != x`): the proxy's.
    template <typename U>
        requires(!element_ref_detail::is_element_ref<std::remove_cvref_t<U>>) &&
                requires(const P& p, U&& u) { p == std::forward<U>(u); }
    [[nodiscard]] friend CHUNK_FORCE_INLINE bool operator==(ElementRef&& lhs, U&& rhs)
    {
        return std::as_const(lhs.p_) == std::forward<U>(rhs);
    }
    /// @brief Refused: comparison of a named element proxy.
    template <typename U>
        requires(!element_ref_detail::is_element_ref<std::remove_cvref_t<U>>) &&
                requires(const P& p, U&& u) { p == std::forward<U>(u); }
    [[nodiscard]] friend bool operator==(ElementRef& /*lhs*/, U&& /*rhs*/)
    {
        static_assert(element_ref_detail::misuse<U>, CHUNK_ELEMENT_REF_MISUSE);
        return false;
    }
    /// @brief `a[i] == b[j]`: the two values (two proxies compare ambiguously
    ///        with the reversed candidates of C++20).
    template <typename Q, typename T2>
    [[nodiscard]] friend CHUNK_FORCE_INLINE bool operator==(ElementRef&& lhs, ElementRef<Q, T2>&& rhs)
    {
        return static_cast<T>(std::move(lhs)) == static_cast<T2>(std::move(rhs));
    }

    // ------------------------------------------------------------- unary
// -a[i], +a[i], ~a[i], !a[i]: the proxy's where it has them (std::complex's
// operator- is a template, which a conversion cannot reach).
#define CHUNK_ELEMENT_REF_UNARY(OP)                                             \
    [[nodiscard]] CHUNK_FORCE_INLINE decltype(auto) operator OP() const&&       \
        requires requires(const P& p) { OP p; }                                 \
    {                                                                           \
        return wrap(OP p_);                                                     \
    }                                                                           \
    [[nodiscard]] decltype(auto) operator OP() const&                           \
        requires requires(const P& p) { OP p; }                                 \
    {                                                                           \
        static_assert(element_ref_detail::misuse<P>, CHUNK_ELEMENT_REF_MISUSE); \
        return wrap(OP p_);                                                     \
    }

    CHUNK_ELEMENT_REF_UNARY(-)
    CHUNK_ELEMENT_REF_UNARY(+)
    CHUNK_ELEMENT_REF_UNARY(~)
    CHUNK_ELEMENT_REF_UNARY(!)
#undef CHUNK_ELEMENT_REF_UNARY

    // ----------------------------------------------- arithmetic with a value
// The binary operators the proxy defines (ProxyOperators: scalar and composite
// operands, both orders), forwarded. An ElementRef on both sides goes through
// the conversions, as two proxies do.
#define CHUNK_ELEMENT_REF_BINARY(OP)                                                              \
    template <typename U>                                                                         \
        requires(!element_ref_detail::is_element_ref<std::remove_cvref_t<U>>) &&                  \
                requires(const P& p, U&& u) { p OP std::forward<U>(u); }                          \
    [[nodiscard]] friend CHUNK_FORCE_INLINE decltype(auto) operator OP(ElementRef&& lhs, U&& rhs) \
    {                                                                                             \
        return std::as_const(lhs.p_) OP std::forward<U>(rhs);                                     \
    }                                                                                             \
    template <typename U>                                                                         \
        requires(!element_ref_detail::is_element_ref<std::remove_cvref_t<U>>) &&                  \
                requires(const P& p, U&& u) { std::forward<U>(u) OP p; }                          \
    [[nodiscard]] friend CHUNK_FORCE_INLINE decltype(auto) operator OP(U&& lhs, ElementRef&& rhs) \
    {                                                                                             \
        return std::forward<U>(lhs) OP std::as_const(rhs.p_);                                     \
    }

    CHUNK_ELEMENT_REF_BINARY(+)
    CHUNK_ELEMENT_REF_BINARY(-)
    CHUNK_ELEMENT_REF_BINARY(*)
    CHUNK_ELEMENT_REF_BINARY(/)
#undef CHUNK_ELEMENT_REF_BINARY

    /// @brief `os << a[i]`: the proxy's.
    template <typename C, typename Tr>
        requires requires(std::basic_ostream<C, Tr>& os, const P& p) { os << p; }
    friend std::basic_ostream<C, Tr>& operator<<(std::basic_ostream<C, Tr>& os, ElementRef&& r)
    {
        return os << std::as_const(r.p_);
    }

    /// @brief The proxy, for element_ref_detail::unwrap (a temporary only).
    [[nodiscard]] CHUNK_FORCE_INLINE P& proxy() && noexcept
    {
        return p_;
    }

  private:
    /// @brief An operator's result: a proxy comes back as an ElementRef (so
    ///        `auto v = (a[i] += x);` is refused too), anything else as is.
    template <typename R>
    CHUNK_FORCE_INLINE static decltype(auto) wrap(R&& r)
    {
        if constexpr (std::is_same_v<std::remove_cvref_t<R>, P>)
        {
            return ElementRef(P(std::forward<R>(r)));
        }
        else
        {
            return static_cast<std::remove_cvref_t<R>>(std::forward<R>(r)); // a value, never a reference to a temporary
        }
    }

    P p_; ///< The proxy.
};

#undef CHUNK_ELEMENT_REF_MISUSE

} // namespace shrink_array
