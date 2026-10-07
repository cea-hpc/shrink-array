
#pragma once

#include <bit> // std::bit_ceil (C++20)

/**
 * @brief Smallest power of two >= @p value; 1 for 0 and 1.
 * @details std::bit_ceil, which is defined for 0 and 1 (both give 1). The
 *          slab-size rule, TCSlab::slab_bytes_for(), rounds with it.
 */
[[nodiscard]] inline unsigned long NextPowerOf2(unsigned long value) noexcept
{
    return std::bit_ceil(value);
}
