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

// module_api: the names chunk/modules/shrink_array.cppm exports from the
// headers it reads in its purview -- chunk exchange, chunk-size advisor,
// re-chunk decider, mdspan views and tile operations -- each used through
// `import shrink_array;`: they are exported, not only reachable, and their
// templates instantiate in an importer. Exit status 0 when every check holds.

#include <cstring> // memcpy, for GCC 16 (see getting_started_module.cpp)

import std;
import shrink_array;

namespace sa = shrink_array;
namespace md = shrink_array::md;
using shrink_array::println;

namespace
{
int failures = 0;

void check(bool ok, std::string_view what)
{
    if (!ok)
    {
        println("FAILED: {}", what);
        ++failures;
    }
}

// Chunk exchange: the whole array, then one chunk, into fresh arrays.
void exchange(int count)
{
    sa::zero_any::Array<double> src(count);
    for (int i = 0; i < count / 4; ++i)
    {
        src[i] = 1.0 + static_cast<double>(i % 5); // varied: materialised chunks
    }
    for (int i = count / 4; i < count / 2; ++i)
    {
        src[i] = 2.5; // constant: second-value chunks after shrink_to_fit
    }
    src.shrink_to_fit();

    const sa::ArrayState<double> state = sa::export_array(src);
    sa::zero_any::Array<double>  dst(count);
    check(sa::import_array(dst, state), "import_array accepts a fresh array");
    const auto& cs   = std::as_const(src);
    const auto& cd   = std::as_const(dst);
    bool        same = true;
    for (int i = 0; i < count; ++i)
    {
        same = same && cs[i] == cd[i];
    }
    check(same, "import_array gives the source's values");

    const sa::ChunkState<double> first = sa::export_chunk(src, 0);
    check(first.kind == sa::ChunkKind::materialised, "chunk 0 is materialised");
    sa::zero_any::Array<double> one(count);
    check(sa::import_chunk(one, 0, first), "import_chunk accepts a fresh chunk");
    check(std::as_const(one)[3] == cs[3], "import_chunk gives the chunk's values");
}

// Advisor and decider: a report on the array, observed once.
void advisor(int count)
{
    sa::zero_any::Array<double> arr(count);
    for (int i = 0; i < count / 2; ++i)
    {
        arr[i] = static_cast<double>(i % 3);
    }
    const sa::ChunkSizeReport rep = sa::ChunkSizeAdvisor<double>::analyze(arr);
    check(rep.current_size == arr.chunk_size(), "the advisor reports the array's chunk size");
    check(!rep.table.empty(), "the advisor evaluates candidate sizes");
    sa::RechunkDecider decider(sa::RechunkPolicy{});
    (void)decider.observe(&arr, 0, rep, "advisor");
}

// mdspan: a tiled view, a box filled, read back by tile and by box.
void views()
{
    constexpr int               n    = 64;
    const auto                  size = static_cast<int>(md::tiled_size<16, 16>(n, n));
    sa::zero_any::Array<double> arr(size);
    auto                        f = md::tiled_view<16, 16>(arr, n, n);
    md::fill_box(f, std::array<long, 2>{8, 8}, std::array<long, 2>{40, 24}, 4.0);

    const auto cf  = md::as_const(f);
    double     sum = 0.0;
    md::for_each_tile(
        cf, 0, md::tile_count(cf),
        [&](const auto& tl, auto b)
        {
            for (long i = 0; i < tl.extent[0]; ++i)
            {
                for (long j = 0; j < tl.extent[1]; ++j)
                {
                    sum += b[i, j];
                }
            }
        },
        [&](const auto& tl, double v) { sum += v * static_cast<double>(tl.extent[0] * tl.extent[1]); });
    check(sum == 4.0 * 32 * 16, "for_each_tile sums the filled box");

    std::vector<double> box(32 * 16);
    md::copy_box_out(cf, std::array<long, 2>{8, 8}, std::array<long, 2>{40, 24}, box.data());
    check(std::ranges::all_of(box, [](double v) { return v == 4.0; }), "copy_box_out reads the filled box");
    check(cf[0, 0] == 0.0 && cf[8, 8] == 4.0, "the view reads the array");
    println("mdspan: {}", md::mdspan_source);
}
} // namespace

int main()
{
    sa::ParamChunk::init_indexes(true);
    {
        exchange(1 << 16);
        advisor(1 << 16);
        views();
    }
    sa::release_chunk_pools();
    sa::destroy_block_chunk();
    println("module_api: {} failure(s)", failures);
    return failures == 0 ? 0 : 1;
}
