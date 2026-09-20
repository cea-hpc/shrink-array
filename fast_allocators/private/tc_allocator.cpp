/*
 * Copyright 2024 CEA (www.cea.fr)
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

#include "fast_allocators/tc_allocator.h"
#include "fast_allocators/internal/static_const_array.h"

#ifndef _OPENMP // openmp problem allocator remembers itself
                // --> deadlock
void* TCAllocator::operator new(size_t s)
{
    return (void*)TypeAllocator<TCAllocator>::alloc(s);
}

void TCAllocator::operator delete(void* p)
{
    TypeAllocator<TCAllocator>::free(p);
}
#endif
#if defined(DEBUG) || defined(_OPENMP)
bool __debug_mem = false;
#endif