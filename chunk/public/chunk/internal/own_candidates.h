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

// A pass over one array's candidates only.
//
// resize() must empty the candidate lists of the chunks of the array it
// resizes: they hold &chunk_list_[j] pointers, which its reallocation would
// leave dangling. At a sequential point it runs the pass on every rank's whole
// list. Inside a region -- an array resized by the thread that owns it -- it
// ran the pass on the calling thread's whole list, with no writer handshake
// (no proxy): right for the array being resized, which no other thread writes,
// and wrong for every other array that thread had queued a chunk of. A shared
// array's chunk, being written by another thread at that moment, was scanned
// and compressed under its writer.
//
// with_own_candidates_only() sets the other arrays' candidates aside for the
// time of the pass and puts them back after it: the pass is the same, on the
// chunks it has the right to.

#include <algorithm> // std::stable_partition
#include <iterator>  // std::make_move_iterator
#include <tuple>     // std::get
#include <vector>    // std::vector

namespace shrink_array::internal
{

/**
 * @brief Run @p pass with @p list reduced to the candidates of array @p num_tab.
 * @details The candidates of the other arrays leave the list before @p pass
 *          and return to its end after it, in their order. @p list is one
 *          rank's own: nobody else appends to it meanwhile.
 * @tparam List A candidate list: tuples whose third element is the num_tab.
 * @tparam Pass Callable, no argument.
 * @param list The calling thread's candidate list.
 * @param num_tab The array whose candidates stay.
 * @param pass The compression pass to run on what stays.
 */
template <typename List, typename Pass>
void with_own_candidates_only(List& list, int num_tab, Pass&& pass)
{
    const auto first_foreign = std::stable_partition(list.begin(), list.end(), [num_tab](const auto& candidate)
                                                     { return std::get<2>(candidate) == num_tab; });
    std::vector<typename List::value_type> aside(std::make_move_iterator(first_foreign),
                                                 std::make_move_iterator(list.end()));
    list.erase(first_foreign, list.end());
    const auto put_back = [&]
    { list.insert(list.end(), std::make_move_iterator(aside.begin()), std::make_move_iterator(aside.end())); };
    try
    {
        pass();
    }
    catch (...)
    {
        put_back();
        throw;
    }
    put_back();
}

} // namespace shrink_array::internal
