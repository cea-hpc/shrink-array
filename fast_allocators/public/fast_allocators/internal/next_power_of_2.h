
#pragma once

#include <bit> // std::bit_ceil (C++20)
#include <cstddef>

// Smallest power of two >= x. Modern replacement for the hand-rolled bit-smear
// (kept in comments below): std::bit_ceil is correct on every 64-bit target
// without the old `#if defined(X86_64)` guard, whose absence silently dropped
// the `>> 32` step and broke large sizes on non-X86_64 builds.
//
// Signature kept as (unsigned long) to match all existing call sites, one of
// which passes a signed int (implicitly converted, as with the legacy version).
// std::bit_ceil is UB for 0 and returns 1 for 1, so inputs <= 1 are clamped to
// 1; every call site passes a block size (element size + header), always >= 2,
// so the clamp is pure defense and never triggers in practice.
[[nodiscard]] inline unsigned long NextPowerOf2(unsigned long value) noexcept
{
    return value <= 1UL ? 1UL : std::bit_ceil(value);
}

// Legacy implementation (pre-C++20), retained for reference:
//
// inline unsigned long NextPowerOf2(unsigned long x)
// {
//     --x;
//     x |= x >> 1;  x |= x >> 2;  x |= x >> 4;
//     x |= x >> 8;  x |= x >> 16;
// #if defined(X86_64)
//     x |= x >> 32;   // <-- silently absent on non-X86_64 builds
// #endif
//     return ++x;
// }
