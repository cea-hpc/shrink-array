#pragma once

/*
 * Copyright 2024 CEA (www.cea.fr)
 * Licensed under the Apache License, Version 2.0.
 */

// Compile-time chunk geometry. Formerly three runtime globals
// (extern unsigned int __chunk_size/__decalage_chunk/__modulo_chunk defined in
// chunk_std.cpp and set once via ParamChunk::set_chunk_size). All consumers only
// ever set them to a compile-time constant (constexpr chunk_size = 64 in every
// example), so making them constexpr removes the extern/.cpp, folds `i >>
// __decalage_chunk` and `i & __modulo_chunk` into immediate shifts/masks on the
// hot path, and keeps a single shared value.
//
// Override at build time:  cmake -DCHUNK_SIZE=128  (must be a power of two).

#include "chunk/internal/func_p2.h" // constexpr log2int

namespace shrink_array
{
#ifndef CHUNK_SIZE
    #define CHUNK_SIZE 64
#endif

inline constexpr unsigned int __chunk_size     = static_cast<unsigned int>(CHUNK_SIZE);
inline constexpr unsigned int __decalage_chunk = static_cast<unsigned int>(log2int(CHUNK_SIZE));
inline constexpr unsigned int __modulo_chunk   = static_cast<unsigned int>(CHUNK_SIZE) - 1u;

static_assert(CHUNK_SIZE >= 1, "CHUNK_SIZE must be positive");
static_assert((CHUNK_SIZE & (CHUNK_SIZE - 1)) == 0, "CHUNK_SIZE must be a power of two");

} // namespace shrink_array
