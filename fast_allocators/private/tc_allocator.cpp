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

#include <cstddef> // size_t (serial builds only: operator new below)

#include "fast_allocators/tc_allocator.h"                // TCAllocator (serial builds only)
#include "fast_allocators/internal/static_const_array.h" // TypeAllocator (serial builds only)

namespace fast_alloc
{

#if !FAST_ALLOC_PMP_THREADED // serial builds only: threaded, the allocator would re-enter itself -> deadlock
template <>
void* TCSlab<FreeLinks::InPayload>::operator new(size_t size)
{
    return TypeAllocator<TCAllocator>::alloc(size);
}

template <>
void TCSlab<FreeLinks::InPayload>::operator delete(void* ptr)
{
    TypeAllocator<TCAllocator>::free(ptr);
}
#endif

} // namespace fast_alloc
