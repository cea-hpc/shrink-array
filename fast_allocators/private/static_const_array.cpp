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

#include <cstdlib> // size_t
#include <set>     // the registry of allocator instances, filled in serial builds only
#include <tuple>   // std::apply (the ladder report)

#include "fast_allocators/internal/singleton.h"          // Singleton
#include "fast_allocators/tc_allocator.h"                // VTCTypeAllocator, TCTypeAllocatorEBC
#include "fast_allocators/internal/alloc_config.h"       // alloc_config::use_std_alloc
#include "fast_allocators/internal/static_const_array.h" // the declarations defined here
#include "fast_allocators/internal/pmp_alloc.h"          // FAST_ALLOC_PMP_MAX_THREADS
#include "fast_allocators/internal/svalloc.h"            // svalloc, svcalloc, svfree

namespace fast_alloc
{

/// @copydoc InfoFreeElt::InfoFreeElt
/// @note Sized by FAST_ALLOC_PMP_MAX_THREADS at construction (before main). A rank past
///       it -- a team raised later -- is not batched: try_free_block() frees its
///       blocks at once rather than index past these arrays.
InfoFreeElt::InfoFreeElt()
{
    // Serially there is no batching (try_free_block never enters the batching
    // branch), so the 2 x 10000-entry arrays are not worth allocating.
    if constexpr (alloc_config::use_openmp)
    {
        const int max_thr = FAST_ALLOC_PMP_MAX_THREADS;

        tab_sz  = svalloc<IntMaxFreeElt>(max_thr);
        tab_ptr = svalloc<VoidMaxFreeElt>(max_thr);

        compt    = svcalloc<int>(max_thr);
        somme_sz_ = svcalloc<int>(max_thr);

        nb_thr = max_thr;
    }
}

/// @copydoc InfoFreeElt::~InfoFreeElt
InfoFreeElt::~InfoFreeElt()
{
    if (tab_ptr == nullptr)
    {
        return; // serial build: nothing was allocated
    }
    svfree(tab_ptr);
    svfree(tab_sz);

    svfree(compt);
    svfree(somme_sz_);

    // An ordinary global, unlike the ladder and the arenas, which are never
    // destroyed: a ladder free in a region, or a parallel_end flush, may still
    // come after this destructor (another library's static or TLS destructor,
    // the OpenMP runtime's shutdown). Leave an empty batch set behind, not
    // dangling pointers: with nb_thr 0, try_free_block() frees at once and
    // clear_info_free_elt() has nothing to walk.
    tab_ptr  = nullptr;
    tab_sz   = nullptr;
    compt    = nullptr;
    somme_sz_ = nullptr;
    nb_thr   = 0;
}

/// @copydoc StaticConstArray::clear_info_free_elt(int)
void StaticConstArray::clear_info_free_elt(int thr_rank)
{
    for (int k = 0; k < tab_info_free_elt().compt[thr_rank]; ++k)
    {
        const int szl  = tab_info_free_elt().tab_sz[thr_rank][k];
        void*     ptrl = tab_info_free_elt().tab_ptr[thr_rank][k];
        free_block_with_int(szl, ptrl);
    }
}

/// @copydoc StaticConstArray::clear_info_free_elt()
void StaticConstArray::clear_info_free_elt()
{
    // Over the ranks the batches were SIZED for, not FAST_ALLOC_PMP_MAX_THREADS: that may
    // have grown since (omp_set_num_threads) and would index past the arrays;
    // nb_thr is 0 serially, where nothing was allocated. reset(), not compt = 0:
    // somme_sz must come down too, or the emptied batch still counts as full.
    for (int thr_rank = 0; thr_rank < tab_info_free_elt().nb_thr; ++thr_rank)
    {
        clear_info_free_elt(thr_rank);
        tab_info_free_elt().reset(thr_rank);
    }
}

/**
 * @brief Registry of the live ::TCTypeAllocator instances, for the memory report.
 *
 * @details They are of every element type, so they are held through the
 *          type-erased ::VTCTypeAllocator and each one enrols itself from its
 *          constructor.
 *
 * @warning Serial builds only. An OpenMP build declares no set and every
 *          method below is a no-op, because a std::set with no lock would be
 *          written concurrently by every rank that creates an allocator -- so
 *          compute_sum_allocated_size_tc_alloc() reports the ladder alone there.
 */
class ManageTCTypeAllocator
{
  public:
    friend class Singleton<ManageTCTypeAllocator>;

    ManageTCTypeAllocator()  = default;
    ~ManageTCTypeAllocator() = default;

    // A singleton: one registry per process.
    ManageTCTypeAllocator(const ManageTCTypeAllocator&)            = delete;
    ManageTCTypeAllocator& operator=(const ManageTCTypeAllocator&) = delete;
    ManageTCTypeAllocator(ManageTCTypeAllocator&&)                 = delete;
    ManageTCTypeAllocator& operator=(ManageTCTypeAllocator&&)      = delete;

    /// @brief Enrol an allocator. No-op in an OpenMP build.
    void add_tc_type_allocator_inst([[maybe_unused]] VTCTypeAllocator* ptr_tcallocator)
    {
        if constexpr (!alloc_config::use_openmp)
        {
            set_allocator_.insert(ptr_tcallocator); // serial builds only: no concurrent insert
        }
    }

    /// @brief Withdraw one. No-op in an OpenMP build.
    void remove_tc_type_allocator_inst([[maybe_unused]] VTCTypeAllocator* ptr_tcallocator)
    {
        if constexpr (!alloc_config::use_openmp)
        {
            set_allocator_.erase(ptr_tcallocator); // serial builds only: no concurrent erase
        }
    }

    /// @brief Add every enrolled allocator's footprint to the totals.
    /// @details Adds nothing in an OpenMP build; the set is empty by design.
    void add_allocated_size(size_t& sum_used_size, size_t& sum_nn_ref, size_t& alloc_sz) const
    {
        for (const VTCTypeAllocator* inst : set_allocator_)
        {
            inst->add_allocated_size(sum_used_size, sum_nn_ref, alloc_sz);
        }
    }

  protected:
    // Filled in a serial build only (no concurrent insert); left empty under
    // OpenMP, where the allocators are not enrolled.
    using SetAllocatorT = std::set<VTCTypeAllocator*>;

    SetAllocatorT set_allocator_;
};


void add_tc_type_allocator_inst(VTCTypeAllocator* inst)
{
    Singleton<ManageTCTypeAllocator>::get_ref().add_tc_type_allocator_inst(inst);
}

void remove_tc_type_allocator_inst(VTCTypeAllocator* inst)
{
    Singleton<ManageTCTypeAllocator>::get_ref().remove_tc_type_allocator_inst(inst);
}

/// @copydoc ::compute_sum_allocated_size_tc_alloc
void compute_sum_allocated_size_tc_alloc(size_t& sum_used_size, size_t& sum_nn_ref, size_t& alloc_sz)
{
    sum_used_size = 0;
    sum_nn_ref   = 0;
    alloc_sz     = 0;

    // The ladder exists as a set of objects whatever the build, but under
    // FAST_ALLOC_STD_ALLOC nothing was ever allocated from it, so there is nothing to add.
    if constexpr (alloc_config::use_slabs)
    {
        std::apply([&](const auto&... bucket)
                   { (bucket.add_allocated_size(sum_used_size, sum_nn_ref, alloc_sz), ...); },
                   StaticConstArray::buckets());
    }

    // Not part of StaticConstArray, but counted in the same total.
    Singleton<ManageTCTypeAllocator>::get_ref().add_allocated_size(sum_used_size, sum_nn_ref, alloc_sz);
}

} // namespace fast_alloc
