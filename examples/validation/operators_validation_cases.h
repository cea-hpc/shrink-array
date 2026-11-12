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

/**
 * @file operators_validation_cases.h
 * @brief operators_validation: every operator of an element, a[i], on every
 *        element type, against plain T lvalues -- the cases.
 *
 * @details
 *   The reference is an array of plain T: a[i] stands for a T&, and a plain
 *   T& is exactly what an operator must behave like (std::vector<bool> would
 *   not do: its reference is a proxy of its own). An operator is exercised
 *   wherever the plain T accepts it (a `requires` on T&): one the library
 *   refuses while T accepts it does not compile here.
 *
 *   - compound assignments `+= -= *= /= %= &= |= ^= <<= >>=`, with an operand
 *     of type T, int, unsigned and double (the promotions and truncations of
 *     a mixed operand), one value that changes the elements and one that
 *     keeps or absorbs them (`+= 0`, `*= 0`, `&= 0`), and with another
 *     array's element as the operand; the value each returns as well;
 *   - `++` and `--`, prefix and postfix, and the value each returns;
 *   - `=` from a value of each operand type and from another array's element;
 *   - the unary `- + ~ !` and the binary `+ - * /`, the element on either
 *     side, stored back;
 *   - `==` and `!=` both ways and between two elements, the conversions to T
 *     and to bool, `os << a[i]`.
 *
 *   Each case starts from the same content, eight chunks: two at the first
 *   constant (one never written), two at a second value, one varied, one half
 *   at the first constant, and two that repeat one motif at two offsets
 *   (pattern compresses those to its motif state). A compression pass puts
 *   the constant chunks in their compressed states first; the operator then
 *   runs over every element in a parallel region, and the array is compared
 *   with the reference, then compressed and compared again: the chunks an
 *   operator made uniform fold, the others must read back unchanged.
 *
 *   Every case runs twice in a threaded build with the test hooks: directly,
 *   then with one write in three forced down the deferred path
 *   (test_hooks.h), whose replays must leave the same values and return the
 *   same results.
 *
 *   `--type NAME` runs that element type alone (as type_name<T>() spells it:
 *   bool, int, std::complex<double>, ...).
 *
 *   The variant comes from the target's CHUNK_VARIANT_* macro;
 *   OPERATORS_VALIDATION_PATTERN selects pattern::Array (integer types other
 *   than bool) in a zero_any build.
 *
 *   One translation unit per group of element types
 *   (operators_validation_<group>.cpp); operators_validation.cpp holds run().
 *   All the types in one unit took 7.8 GB to compile (GCC 14, -O3), more than
 *   a 16 GB CI runner has room for next to a second harness.
 */

#pragma once

#include <complex>     // std::complex<double>
#include <cstddef>     // std::size_t
#include <memory>      // std::unique_ptr (the reference arrays)
#include <sstream>     // std::ostringstream (os << a[i])
#include <string>      // std::string
#include <type_traits> // std::conditional_t, std::is_same_v, std::is_integral_v
#include <utility>     // std::forward, std::as_const

#include "chunk/internal/chunk_traits.h" // forward decls + default IntType = int
#include "chunk/zero_array.h"            // zero::Array
#include "chunk/zero_one_value_array.h"  // zero_one_value::Array
#include "chunk/zero_any_array.h"        // zero_any::Array
#if defined(OPERATORS_VALIDATION_PATTERN)
    #include "chunk/pattern_array.h" // pattern::Array
#endif
#include "chunk/config.h"             // chunk_config::use_zero*, build_config::use_openmp, has_not_ompt
#include "chunk/manage_ompt.h"        // begin_parallel_phase / finalize_parallel_phase
#include "chunk/param_chunk.h"        // ParamChunk::init_indexes
#include "chunk/internal/type_name.h" // type_name<T>()
#include "chunk/internal/concept_is_clean_assignement.h" // IsCleanAssignment, IsCleanShiftOp
#include "chunk/print.h"                                 // shrink_array::println
#include "chunk/internal/test_hooks.h"                   // force_defer_every, force_defer_compiled

namespace validation::operators_validation
{

using namespace shrink_array;

#if defined(OPERATORS_VALIDATION_PATTERN)
template <typename T>
using Arr                    = pattern::Array<T>;
constexpr bool pattern_build = true;
#else
template <typename T>
using Arr =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;
constexpr bool pattern_build = false;
#endif

constexpr const char* variant_name()
{
    if constexpr (pattern_build)
    {
        return "pattern";
    }
    else if constexpr (chunk_config::use_zero_any)
    {
        return "zero_any";
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        return "zero_one_value";
    }
    else
    {
        return "zero";
    }
}

constexpr int n_chunks = 8; ///< the layout below, one chunk kind each

inline std::string only_type; ///< --type NAME: that element type alone (type_name<T>(), e.g. bool)

inline void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}

inline void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

// ── the values ──────────────────────────────────────────────────────────────

/// @brief A small value of T from an integer: within [-11, 12], so that one
///        operator on it overflows no type (char included).
template <typename T>
T value_of(long v)
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return v % 3 == 0;
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        return {static_cast<double>(v), static_cast<double>(1 - v % 2)};
    }
    else if constexpr (std::is_unsigned_v<T>)
    {
        return static_cast<T>(v < 0 ? -v : v);
    }
    else
    {
        return static_cast<T>(v);
    }
}

/// @brief The second value of the constant chunks.
template <typename T>
T second_value()
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return true;
    }
    else
    {
        return value_of<T>(5);
    }
}

/// @brief The starting content, by chunk: 0 and 1 at T{} (1 never written),
///        2 and 3 at the second value, 4 varied, 5 half at T{}, 6 and 7 one
///        motif at two offsets.
template <typename T>
T start_value(long i, long chunk_sz)
{
    const long j    = i % chunk_sz;
    const auto base = static_cast<long>((j * 7) % 23) - 11;
    switch (i / chunk_sz)
    {
        case 0:
        case 1:
            return T{};
        case 2:
        case 3:
            return second_value<T>();
        case 4:
            return value_of<T>(((i * 5) % 23) - 11);
        case 5:
            return j % 2 ? T{} : value_of<T>(base);
        case 6:
            return value_of<T>(base);
        default:
            return value_of<T>(base + 1);
    }
}

/// @brief The other array's content (the element operands): never T{}, so it
///        divides and shifts; small, so it shifts by at most 3. Its chunk 0 is
///        constant.
template <typename T>
T operand_value(long i, long chunk_sz)
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return true;
    }
    else
    {
        return i < chunk_sz ? value_of<T>(1) : value_of<T>(1 + (i * 5) % 3);
    }
}

// ── the arrays and their reference ──────────────────────────────────────────

/// @brief An array and its reference of plain T, with the same content.
template <typename T>
struct Pair
{
    Arr<T>               arr;
    std::unique_ptr<T[]> ref;
    long                 count;

    explicit Pair(long n) : arr(n), ref(new T[static_cast<std::size_t>(n)]()), count(n) {}
};

/// @brief Write @p value_at into both sides of @p pair, then one compression
///        pass, so that the constant chunks start compressed. The T{} of
///        chunk 1 stay unwritten (a fresh array already reads them).
template <typename T, typename F>
void fill(Pair<T>& pair, long chunk_sz, F value_at)
{
    if constexpr (requires(Arr<T> probe, T value) { probe.set_second_value(value); })
    {
        pair.arr.set_second_value(second_value<T>(), false);
    }
    for (long i = 0; i < pair.count; ++i)
    {
        const T v                             = value_at(i, chunk_sz);
        pair.ref[static_cast<std::size_t>(i)] = v;
        if (i / chunk_sz != 1 || !(v == T{}))
        {
            pair.arr[i] = v;
        }
    }
#pragma omp parallel
    {
        begin_phase();
        pair.arr.compress();
    }
    end_phase();
}

/// @brief A value as a message shows it: a char as its number.
template <typename T>
auto printable(const T& value)
{
    if constexpr (std::is_integral_v<T> && sizeof(T) < sizeof(int))
    {
        return static_cast<int>(value);
    }
    else
    {
        return value;
    }
}

/// @brief Compare the array with its reference, element by element.
/// @return the mismatches (the first few printed).
template <typename T>
long compare(const Pair<T>& pair, const std::string& what)
{
    long bad = 0;
    for (long i = 0; i < pair.count; ++i)
    {
        const T got  = static_cast<T>(std::as_const(pair.arr)[i]);
        const T want = pair.ref[static_cast<std::size_t>(i)];
        if (!(got == want))
        {
            if (bad < 4)
            {
                std::ostringstream os;
                os << "got " << printable(got) << ", want " << printable(want);
                println("[{}] FAIL {}: element {}: {}", variant_name(), what, i, os.str());
            }
            ++bad;
        }
    }
    return bad;
}

/// @brief Compare, compress, compare again.
template <typename T>
long check(Pair<T>& pair, const std::string& what)
{
    const long bad = compare(pair, what);
#pragma omp parallel
    {
        begin_phase();
        pair.arr.compress();
    }
    end_phase();
    return bad + compare(pair, what + " (after a pass)");
}

/// @brief What a case reports: its name, and the results returned.
template <typename T>
long compare_returned(const std::unique_ptr<T[]>& got,
                      const std::unique_ptr<T[]>& want,
                      long                        count,
                      const std::string&          what)
{
    long bad = 0;
    for (long i = 0; i < count; ++i)
    {
        if (!(got[static_cast<std::size_t>(i)] == want[static_cast<std::size_t>(i)]))
        {
            if (bad < 4)
            {
                std::ostringstream os;
                os << "returned " << printable(got[static_cast<std::size_t>(i)]) << ", want "
                   << printable(want[static_cast<std::size_t>(i)]);
                println("[{}] FAIL {}: element {}: {}", variant_name(), what, i, os.str());
            }
            ++bad;
        }
    }
    return bad;
}

// ── the operators ───────────────────────────────────────────────────────────
//
// One struct per operator: its spelling, whether a plain T accepts it with an
// operand of type C, and its application -- the same code on a[i] (an
// ElementRef temporary, forwarded: a named one is refused) and on a plain T&.

#define OPERATORS_COMPOUND(NAME, OP)                                          \
    struct NAME                                                               \
    {                                                                         \
        static constexpr const char* name = #OP;                              \
        template <typename T, typename C>                                     \
        static constexpr bool valid = requires(T& x, const C& c) { x OP c; }; \
        template <typename E, typename C>                                     \
        static decltype(auto) apply(E&& elem, C&& operand)                    \
        {                                                                     \
            return std::forward<E>(elem) OP std::forward<C>(operand);         \
        }                                                                     \
    };
OPERATORS_COMPOUND(AddAssign, +=)
OPERATORS_COMPOUND(SubAssign, -=)
OPERATORS_COMPOUND(MulAssign, *=)
OPERATORS_COMPOUND(DivAssign, /=)
OPERATORS_COMPOUND(ModAssign, %=)
OPERATORS_COMPOUND(AndAssign, &=)
OPERATORS_COMPOUND(OrAssign, |=)
OPERATORS_COMPOUND(XorAssign, ^=)
OPERATORS_COMPOUND(ShlAssign, <<=)
OPERATORS_COMPOUND(ShrAssign, >>=)
OPERATORS_COMPOUND(Assign, =)
#undef OPERATORS_COMPOUND

#define OPERATORS_INC_DEC(NAME, PREFIX, POSTFIX)                            \
    struct NAME                                                             \
    {                                                                       \
        static constexpr const char* name = #PREFIX "x" #POSTFIX;           \
        template <typename T>                                               \
        static constexpr bool valid = requires(T& x) { PREFIX x POSTFIX; }; \
        template <typename E>                                               \
        static decltype(auto) apply(E&& elem)                               \
        {                                                                   \
            return PREFIX std::forward<E>(elem) POSTFIX;                    \
        }                                                                   \
    };
OPERATORS_INC_DEC(PreInc, ++, )
OPERATORS_INC_DEC(PreDec, --, )
OPERATORS_INC_DEC(PostInc, , ++)
OPERATORS_INC_DEC(PostDec, , --)
#undef OPERATORS_INC_DEC

#define OPERATORS_UNARY(NAME, OP)                                                   \
    struct NAME                                                                     \
    {                                                                               \
        static constexpr const char* name = "x = " #OP "x";                         \
        template <typename T>                                                       \
        static constexpr bool valid = requires(T& x) { x = static_cast<T>(OP x); }; \
        template <typename T, typename D, typename S>                               \
        static void apply(D&& dst, S&& src)                                         \
        {                                                                           \
            std::forward<D>(dst) = static_cast<T>(OP std::forward<S>(src));         \
        }                                                                           \
    };
OPERATORS_UNARY(Negate, -)
OPERATORS_UNARY(Plus, +)
OPERATORS_UNARY(Complement, ~)
OPERATORS_UNARY(Not, !)
#undef OPERATORS_UNARY

#define OPERATORS_BINARY(NAME, OP)                                                                   \
    struct NAME                                                                                      \
    {                                                                                                \
        static constexpr const char* name = #OP;                                                     \
        template <typename T, typename C>                                                            \
        static constexpr bool valid = requires(T& x, const C& c) {                                   \
            x = static_cast<T>(x OP c);                                                              \
            x = static_cast<T>(c OP x);                                                              \
        };                                                                                           \
        /* x = x OP c */                                                                             \
        template <typename T, typename D, typename S, typename C>                                    \
        static void apply_left(D&& dst, S&& src, C&& operand)                                        \
        {                                                                                            \
            std::forward<D>(dst) = static_cast<T>(std::forward<S>(src) OP std::forward<C>(operand)); \
        }                                                                                            \
        /* x = c OP x */                                                                             \
        template <typename T, typename D, typename S, typename C>                                    \
        static void apply_right(D&& dst, S&& src, C&& operand)                                       \
        {                                                                                            \
            std::forward<D>(dst) = static_cast<T>(std::forward<C>(operand) OP std::forward<S>(src)); \
        }                                                                                            \
    };
OPERATORS_BINARY(Add, +)
OPERATORS_BINARY(Sub, -)
OPERATORS_BINARY(Mul, *)
OPERATORS_BINARY(Div, /)
#undef OPERATORS_BINARY

// ── the cases ───────────────────────────────────────────────────────────────

/// @brief The context of a run: the element count and the chunk size, the
///        labels, and the failures so far.
struct Run
{
    long        count;
    long        chunk_sz;
    const char* type;
    const char* mode;
    long        fails = 0;
    long        cases = 0;

    std::string label(const char* op, const char* operand) const
    {
        return std::string(type) + " " + op + " " + operand + " [" + mode + "]";
    }
};

/// @brief The compound assignments an array of bool refuses at compile time
///        (ElementArithmetic, concept_is_clean_assignement.h), where a plain
///        bool takes them: arithmetic on a truth value. Not run; the
///        static_asserts of refuses_bool_arithmetic() check the refusal.
template <typename Op>
constexpr bool refused_on_bool =
    std::is_same_v<Op, AddAssign> || std::is_same_v<Op, SubAssign> || std::is_same_v<Op, MulAssign> ||
    std::is_same_v<Op, DivAssign> || std::is_same_v<Op, ModAssign> || std::is_same_v<Op, ShlAssign> ||
    std::is_same_v<Op, ShrAssign>;

/// @brief Left out on a bool: `*` and `~`, valid and warned about as on a
///        plain bool (-Wint-in-bool-context, -Wbool-operation), and the
///        refused_on_bool assignments.
template <typename Op, typename T>
constexpr bool bool_warning =
    std::is_same_v<T, bool> && (std::is_same_v<Op, Mul> || std::is_same_v<Op, Complement> || refused_on_bool<Op>);

/// @brief `a[i] OP= 1` does not compile on an array of B (bool), for every
///        refused_on_bool operator; `=`, `&=`, `|=` and `^=` do.
template <typename B>
void refuses_bool_arithmetic()
{
    static_assert(!requires(Arr<B>& arr) { arr[0] += 1; });
    static_assert(!requires(Arr<B>& arr) { arr[0] -= 1; });
    static_assert(!requires(Arr<B>& arr) { arr[0] *= 1; });
    static_assert(!requires(Arr<B>& arr) { arr[0] /= 1; });
    static_assert(!requires(Arr<B>& arr) { arr[0] %= 1; });
    static_assert(!requires(Arr<B>& arr) { arr[0] <<= 1; });
    static_assert(!requires(Arr<B>& arr) { arr[0] >>= 1; });
    static_assert(requires(Arr<B>& arr) {
        arr[0] = true;
        arr[0] &= true;
        arr[0] |= true;
        arr[0] ^= true;
    });
}

/// @brief Does the library take a C operand of `T OP=` as a plain T does?
/// @details The clean operands (concept_is_clean_assignement.h): the same
///          type, an integer into a floating or complex T, int into a small
///          integer, any integer as a shift or a bitwise operand. The others
///          compile with an HPC warning ([[deprecated]]) and convert the
///          operand to T first -- `int x *= 2.5` multiplies by 2 -- by design,
///          unlike a plain T: not compared here.
template <typename Op, typename T, typename C>
constexpr bool clean_operand()
{
    if constexpr (std::is_same_v<Op, ShlAssign> || std::is_same_v<Op, ShrAssign>)
    {
        return IsCleanShiftOp<T, C>;
    }
    else if constexpr (std::is_same_v<Op, AndAssign> || std::is_same_v<Op, OrAssign> || std::is_same_v<Op, XorAssign>)
    {
        return std::is_integral_v<T> && std::is_integral_v<C>;
    }
    else
    {
        return IsCleanAssignment<T, C>;
    }
}

/// @brief `a[i] OP= c` on every element, and the value each returns, for an
///        operand of type C. A divisor or modulus that is 0 is not run.
template <typename Op, typename T, typename C>
void compound_scalar(Run& run, const char* operand_name, C operand)
{
    if constexpr (Op::template valid<T, C> && clean_operand<Op, T, C>() && !bool_warning<Op, T>)
    {
        if constexpr (std::is_same_v<Op, DivAssign> || std::is_same_v<Op, ModAssign>)
        {
            if (operand == C{})
            {
                return;
            }
        }
        Pair<T>              pair(run.count);
        std::unique_ptr<T[]> got(new T[static_cast<std::size_t>(run.count)]());
        std::unique_ptr<T[]> want(new T[static_cast<std::size_t>(run.count)]());
        fill(pair, run.chunk_sz, &start_value<T>);
#pragma omp parallel
        {
            begin_phase();
#pragma omp for
            for (long i = 0; i < run.count; ++i)
            {
                got[static_cast<std::size_t>(i)] = static_cast<T>(Op::apply(pair.arr[i], operand));
            }
        }
        end_phase();
        for (long i = 0; i < run.count; ++i)
        {
            want[static_cast<std::size_t>(i)] =
                static_cast<T>(Op::apply(pair.ref[static_cast<std::size_t>(i)], operand));
        }
        const std::string what = run.label(Op::name, operand_name);
        run.fails += check(pair, what) + compare_returned(got, want, run.count, what);
        ++run.cases;
    }
}

/// @brief `a[i] OP= b[i]`: the operand another array's element.
template <typename Op, typename T>
void compound_element(Run& run)
{
    if constexpr (Op::template valid<T, T> && !bool_warning<Op, T>)
    {
        Pair<T> pair(run.count);
        Pair<T> other(run.count);
        fill(pair, run.chunk_sz, &start_value<T>);
        fill(other, run.chunk_sz, &operand_value<T>);
#pragma omp parallel
        {
            begin_phase();
#pragma omp for
            for (long i = 0; i < run.count; ++i)
            {
                Op::apply(pair.arr[i], other.arr[i]);
            }
        }
        end_phase();
        for (long i = 0; i < run.count; ++i)
        {
            Op::apply(pair.ref[static_cast<std::size_t>(i)], other.ref[static_cast<std::size_t>(i)]);
        }
        run.fails += check(pair, run.label(Op::name, "b[i]")) + compare(other, run.label(Op::name, "b[i], b"));
        ++run.cases;
    }
}

/// @brief One compound operator with every operand: T, int, unsigned and
///        double values, then an element. @p change alters the elements,
///        @p keep keeps or absorbs them.
template <typename Op, typename T>
void compound(Run& run, long change, long keep)
{
    for (const long v : {change, keep})
    {
        compound_scalar<Op, T>(run, "T", value_of<T>(v));
        if constexpr (!std::is_same_v<T, int>)
        {
            compound_scalar<Op, T>(run, "int", static_cast<int>(v));
        }
        if constexpr (!std::is_same_v<T, unsigned>)
        {
            compound_scalar<Op, T>(run, "unsigned", static_cast<unsigned>(v));
        }
        if constexpr (!std::is_same_v<T, double>)
        {
            compound_scalar<Op, T>(run, "double", static_cast<double>(v));
        }
    }
    compound_scalar<Op, T>(run, "double 2.5", 2.5); // a fraction: no integer rounding hides it
    compound_element<Op, T>(run);
}

/// @brief `++a[i]`, `a[i]--`, ...: the elements and the values returned.
template <typename Op, typename T>
void inc_dec(Run& run)
{
    if constexpr (Op::template valid<T>)
    {
        Pair<T>              pair(run.count);
        std::unique_ptr<T[]> got(new T[static_cast<std::size_t>(run.count)]());
        std::unique_ptr<T[]> want(new T[static_cast<std::size_t>(run.count)]());
        fill(pair, run.chunk_sz, &start_value<T>);
#pragma omp parallel
        {
            begin_phase();
#pragma omp for
            for (long i = 0; i < run.count; ++i)
            {
                got[static_cast<std::size_t>(i)] = static_cast<T>(Op::apply(pair.arr[i]));
            }
        }
        end_phase();
        for (long i = 0; i < run.count; ++i)
        {
            want[static_cast<std::size_t>(i)] = static_cast<T>(Op::apply(pair.ref[static_cast<std::size_t>(i)]));
        }
        const std::string what = run.label(Op::name, "");
        run.fails += check(pair, what) + compare_returned(got, want, run.count, what);
        ++run.cases;
    }
}

/// @brief `a[i] = OP a[i]`.
template <typename Op, typename T>
void unary(Run& run)
{
    if constexpr (Op::template valid<T> && !bool_warning<Op, T>)
    {
        Pair<T> pair(run.count);
        fill(pair, run.chunk_sz, &start_value<T>);
#pragma omp parallel
        {
            begin_phase();
#pragma omp for
            for (long i = 0; i < run.count; ++i)
            {
                Op::template apply<T>(pair.arr[i], pair.arr[i]);
            }
        }
        end_phase();
        for (long i = 0; i < run.count; ++i)
        {
            T& x = pair.ref[static_cast<std::size_t>(i)];
            Op::template apply<T>(x, x);
        }
        run.fails += check(pair, run.label(Op::name, ""));
        ++run.cases;
    }
}

/// @brief `a[i] = a[i] OP c`, `a[i] = c OP a[i]`, and `a[i] = a[i] OP b[i]`.
template <typename Op, typename T, typename C>
void binary(Run& run, const char* operand_name, C operand)
{
    if constexpr (Op::template valid<T, C> && !bool_warning<Op, T>)
    {
        for (const bool left : {true, false})
        {
            if constexpr (std::is_same_v<Op, Div>)
            {
                if (left && operand == C{})
                {
                    continue;
                }
            }
            Pair<T> pair(run.count);
            // c / a[i] divides by the element: start from the operand array, never T{}.
            if (std::is_same_v<Op, Div> && !left)
            {
                fill(pair, run.chunk_sz, &operand_value<T>);
            }
            else
            {
                fill(pair, run.chunk_sz, &start_value<T>);
            }
#pragma omp parallel
            {
                begin_phase();
#pragma omp for
                for (long i = 0; i < run.count; ++i)
                {
                    if (left)
                    {
                        Op::template apply_left<T>(pair.arr[i], pair.arr[i], operand);
                    }
                    else
                    {
                        Op::template apply_right<T>(pair.arr[i], pair.arr[i], operand);
                    }
                }
            }
            end_phase();
            for (long i = 0; i < run.count; ++i)
            {
                T& x = pair.ref[static_cast<std::size_t>(i)];
                if (left)
                {
                    Op::template apply_left<T>(x, x, operand);
                }
                else
                {
                    Op::template apply_right<T>(x, x, operand);
                }
            }
            run.fails += check(pair, run.label(left ? "x = x op c" : "x = c op x", Op::name) + " " + operand_name);
            ++run.cases;
        }
    }
    if constexpr (Op::template valid<T, T> && !bool_warning<Op, T>)
    {
        if (std::string(operand_name) == "T")
        {
            Pair<T> pair(run.count);
            Pair<T> other(run.count);
            fill(pair, run.chunk_sz, &start_value<T>);
            fill(other, run.chunk_sz, &operand_value<T>);
#pragma omp parallel
            {
                begin_phase();
#pragma omp for
                for (long i = 0; i < run.count; ++i)
                {
                    Op::template apply_left<T>(pair.arr[i], pair.arr[i], other.arr[i]);
                }
            }
            end_phase();
            for (long i = 0; i < run.count; ++i)
            {
                T& x = pair.ref[static_cast<std::size_t>(i)];
                Op::template apply_left<T>(x, x, other.ref[static_cast<std::size_t>(i)]);
            }
            run.fails += check(pair, run.label("x = x op b[i]", Op::name));
            ++run.cases;
        }
    }
}

/// @brief The reads: `==` and `!=` both ways and between two elements, the
///        conversions to T and to bool, `os << a[i]`. Counted on both sides.
template <typename T>
void reads(Run& run)
{
    Pair<T> pair(run.count);
    Pair<T> other(run.count);
    fill(pair, run.chunk_sz, &start_value<T>);
    fill(other, run.chunk_sz, &operand_value<T>);

    long bad = 0;
    for (const T c : {T{}, second_value<T>(), value_of<T>(1)})
    {
        long got_eq = 0, want_eq = 0, got_ne = 0, want_ne = 0;
        for (long i = 0; i < run.count; ++i)
        {
            const T& x = pair.ref[static_cast<std::size_t>(i)];
            got_eq += (pair.arr[i] == c) + (c == pair.arr[i]);
            want_eq += (x == c) + (c == x);
            got_ne += (pair.arr[i] != c) + (c != pair.arr[i]);
            want_ne += (x != c) + (c != x);
        }
        bad += (got_eq != want_eq) + (got_ne != want_ne);
    }
    long got_elem = 0, want_elem = 0, got_true = 0, want_true = 0;
    long got_conv = 0;
    for (long i = 0; i < run.count; ++i)
    {
        const T& x = pair.ref[static_cast<std::size_t>(i)];
        const T& y = other.ref[static_cast<std::size_t>(i)];
        got_elem += (pair.arr[i] == other.arr[i]) + (pair.arr[i] != other.arr[i]) * 2;
        want_elem += (x == y) + (x != y) * 2;
        if constexpr (requires(const T& v) { static_cast<bool>(v); })
        {
            got_true += pair.arr[i] ? 1 : 0;
            want_true += x ? 1 : 0;
        }
        const T converted = pair.arr[i];
        got_conv += (converted == x) ? 0 : 1;
    }
    bad += (got_elem != want_elem) + (got_true != want_true) + got_conv;

    std::ostringstream got_os;
    std::ostringstream want_os;
    for (long i = 0; i < run.count; ++i)
    {
        got_os << pair.arr[i] << ' ';
        want_os << pair.ref[static_cast<std::size_t>(i)] << ' ';
    }
    bad += got_os.str() != want_os.str();
    if (bad != 0)
    {
        println("[{}] FAIL {}: {} read check(s) differ", variant_name(), run.label("reads", ""), bad);
    }
    run.fails += bad;
    ++run.cases;
}

/// @brief Every operator on T, in one mode.
template <typename T>
long run_type(long count, long chunk_sz, const char* mode, long& cases)
{
    const std::string type = type_name<T>();
    if (!only_type.empty() && type != only_type)
    {
        return 0;
    }
    Run run{count, chunk_sz, type.c_str(), mode};

    compound<AddAssign, T>(run, 3, 0);
    compound<SubAssign, T>(run, 3, 0);
    compound<MulAssign, T>(run, 2, 0);
    compound<DivAssign, T>(run, 2, 1);
    compound<ModAssign, T>(run, 3, 7);
    compound<AndAssign, T>(run, 6, 0);
    compound<OrAssign, T>(run, 1, 0);
    compound<XorAssign, T>(run, 3, 0);
    compound<ShlAssign, T>(run, 1, 0);
    compound<ShrAssign, T>(run, 1, 0);
    compound<Assign, T>(run, 3, 0);

    inc_dec<PreInc, T>(run);
    inc_dec<PreDec, T>(run);
    inc_dec<PostInc, T>(run);
    inc_dec<PostDec, T>(run);

    unary<Negate, T>(run);
    unary<Plus, T>(run);
    unary<Complement, T>(run);
    unary<Not, T>(run);

    binary<Add, T>(run, "T", value_of<T>(3));
    binary<Sub, T>(run, "T", value_of<T>(3));
    binary<Mul, T>(run, "T", value_of<T>(2));
    binary<Div, T>(run, "T", value_of<T>(2));
    binary<Add, T>(run, "int", 3);
    binary<Mul, T>(run, "double", 2.5);

    reads<T>(run);

    if (run.fails == 0)
    {
        println("[{}] ok: {} [{}], {} cases", variant_name(), type, mode, run.cases);
    }
    cases += run.cases;
    return run.fails;
}

/// @brief run_type() over each group of element types, in one mode; one
///        translation unit each. The floating, bool and complex groups run
///        nothing in a pattern build.
long run_small_integers(long count, long chunk_sz, const char* mode, long& cases);
long run_wide_integers(long count, long chunk_sz, const char* mode, long& cases);
long run_floating(long count, long chunk_sz, const char* mode, long& cases);
long run_bool_complex(long count, long chunk_sz, const char* mode, long& cases);

} // namespace validation::operators_validation
