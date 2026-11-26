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
 * @file device_convert_validation.cpp
 * @brief The host <-> device conversion: a host zero_any::Array exported (export_array),
 *        loaded into a device::Array (assign), written and compressed there, brought back
 *        (image, import_array) and compared with a host mirror, element by element and
 *        chunk kind by chunk kind.
 *
 * @details The C++ half; the device half is device_convert_device.cu (see
 *          device_convert_validation.h). Built with -DCHUNK_DEVICE_CORE_CHECK=ON, for CUDA
 *          or HIP. The source has, every five chunks: a chunk never written (first value), a
 *          chunk of 1.5 (a second value after shrink_to_fit), a varied chunk, a chunk of
 *          zeros with one -0.0 (materialised: -0.0 is not 0.0), a half-constant chunk; and a
 *          partial last chunk. On the device: 2.5 over ten whole chunks (they fold back to a
 *          second value, materialised ones included), a(i) = i over a span across four
 *          chunks; then compress(). zero_or_value declares 2.5 its second value after the
 *          first assign(): the chunks of 1.5 get a buffer there (no second value yet), the ten
 *          of 2.5 fold back and image() declares 2.5.
 */

#include <cstdint>   // std::int64_t
#include <cstdio>    // std::printf, std::snprintf
#include <cstring>   // std::memcmp
#include <exception> // std::exception
#include <utility>   // std::as_const
#include <vector>    // std::vector

#include "chunk/chunk_exchange.h"     // export_array, import_array, ChunkKind
#include "chunk/internal/pmp_chunk.h" // pmp::current_team
#include "chunk/phases.h"             // parallel_phase
#include "chunk/runtime.h"            // Runtime::init
#include "chunk/zero_any_array.h"     // zero_any::Array

#include "device_convert_validation.h" // device_round_trip

namespace sa = shrink_array;

namespace
{

using HostArray = sa::zero_any::Array<double>;

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    g_failures += ok ? 0 : 1;
}

bool same_bits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof a) == 0;
}

double source(std::int64_t i, std::int64_t s)
{
    const std::int64_t k = i % s;
    switch ((i / s) % 5)
    {
        case 0:
            return 0.0;
        case 1:
            return 1.5;
        case 2:
            return (static_cast<double>(i) * 0.25) + 1.0;
        case 3:
            return k == 7 ? -0.0 : 0.0;
        default:
            return k < s / 2 ? 3.0 : static_cast<double>(i);
    }
}

/// @brief Elements of @p got that differ from @p want (bitwise).
long wrong(const std::vector<double>& got, const std::vector<double>& want)
{
    long w = got.size() == want.size() ? 0 : 1;
    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i)
    {
        w += same_bits(got[i], want[i]) ? 0 : 1;
    }
    return w;
}

long count_kind(const sa::ArrayState<double>& st, sa::ChunkKind kind, std::size_t used)
{
    long n = 0;
    for (std::size_t c = 0; c < used; ++c)
    {
        n += st.chunks[c].kind == kind ? 1 : 0;
    }
    return n;
}

/// @brief The round trip into a device array of the zero_any variant, the zero one (there
///        the ten chunks of 1.5 are materialised by assign(), and the ten of 2.5 do not fold)
///        or the zero_or_value one (the chunks of 1.5 materialised, those of 2.5 fold).
void run(device_convert::DeviceVariant variant)
{
    using DV                          = device_convert::DeviceVariant;
    const bool        zero_variant    = variant == DV::zero;
    const bool        materialise_1_5 = variant != DV::zero_any;
    const char* const name = variant == DV::zero ? "zero" : variant == DV::zero_or_value ? "zero_or_value" : "zero_any";
    HostArray         src(0);
    const std::int64_t s    = src.chunk_size();
    const std::int64_t n    = (50 * s) + 17;
    const auto         nn   = static_cast<std::size_t>(n);
    const auto         used = static_cast<std::size_t>((n + s - 1) / s);
    src.resize(static_cast<int>(n));
    std::printf("[zero_any::Array<double> <-> device::Array, %s variant] S=%lld n=%lld\n", name,
                static_cast<long long>(s), static_cast<long long>(n));
    sa::parallel_phase(
        [&]
        {
            sa::pmp::current_team().for_static(0, static_cast<int>(n),
                                               [&](int i)
                                               {
                                                   if ((i / s) % 5 != 0)
                                                   {
                                                       src[i] = source(i, s);
                                                   }
                                               });
        });
    src.shrink_to_fit();
    std::vector<double> want(nn);
    for (std::size_t i = 0; i < nn; ++i)
    {
        want[i] = source(static_cast<std::int64_t>(i), s);
    }
    const sa::ArrayState<double> state = sa::export_array(src);
    const long                   mat0  = count_kind(state, sa::ChunkKind::materialised, used);
    char                         line[200];
    std::snprintf(line, sizeof line, "host export: %ld first value, %ld second value, %ld materialised",
                  count_kind(state, sa::ChunkKind::first_value, used),
                  count_kind(state, sa::ChunkKind::second_value, used), mat0);
    expect(count_kind(state, sa::ChunkKind::second_value, used) == 10, line);

    // The device half: assign, two writes, compress, image, assign again.
    const std::int64_t lo1 = 10 * s, hi1 = 20 * s, lo2 = (30 * s) + 5, hi2 = (33 * s) + 9;
    const auto         r = device_convert::device_round_trip(state, variant, lo1, hi1, lo2, hi2);
    // zero, zero_or_value (no second value yet): the second-value chunks get a buffer.
    const long mat_assign = mat0 + (materialise_1_5 ? count_kind(state, sa::ChunkKind::second_value, used) : 0);
    std::snprintf(line, sizeof line, "assign(): %ld wrong, %llu/%ld chunks materialised", wrong(r.after_assign, want),
                  r.materialised_after_assign, mat_assign);
    expect(wrong(r.after_assign, want) == 0 &&
               r.materialised_after_assign == static_cast<unsigned long long>(mat_assign),
           line);

    std::vector<double> mirror = want;
    for (std::int64_t i = lo1; i < hi1; ++i)
    {
        mirror[static_cast<std::size_t>(i)] = 2.5;
    }
    for (std::int64_t i = lo2; i < hi2; ++i)
    {
        mirror[static_cast<std::size_t>(i)] = static_cast<double>(i);
    }

    // Back on the host.
    HostArray dst(0);
    expect(sa::import_array(dst, r.image.state), "import_array() of image()");
    std::vector<double> got(nn);
    for (std::size_t i = 0; i < nn && static_cast<std::int64_t>(dst.size()) == n; ++i)
    {
        got[i] = std::as_const(dst)[static_cast<int>(i)];
    }
    const sa::ArrayState<double> back = sa::export_array(dst);
    std::snprintf(line, sizeof line, "image() -> host: %ld wrong, %ld/%llu chunks materialised", wrong(got, mirror),
                  count_kind(back, sa::ChunkKind::materialised, used), r.materialised_final);
    expect(wrong(got, mirror) == 0 &&
               count_kind(back, sa::ChunkKind::materialised, used) == static_cast<long>(r.materialised_final),
           line);
    bool folded_to_2_5 = true;
    for (std::int64_t c = 10; c < 20; ++c)
    {
        const auto& cs = back.chunks[static_cast<std::size_t>(c)];
        folded_to_2_5 =
            folded_to_2_5 && (zero_variant ? cs.kind == sa::ChunkKind::materialised
                                           : cs.kind == sa::ChunkKind::second_value && same_bits(cs.value, 2.5));
    }
    std::snprintf(line, sizeof line,
                  zero_variant ? "the ten chunks of 2.5 kept their buffers on the device (%llu folds)"
                               : "the ten chunks of 2.5 folded on the device, second value on the host (%llu folds)",
                  r.folded);
    expect(folded_to_2_5, line);
    std::snprintf(line, sizeof line, "assign() over held buffers: %ld wrong", wrong(r.after_reassign, want));
    expect(wrong(r.after_reassign, want) == 0, line);
    expect(r.arena_consistent, "device arena: one buffer per materialised chunk");
    if (variant == DV::zero_or_value)
    {
        expect(r.image.state.has_second_value && same_bits(r.image.state.second_value, 2.5),
               "image(): the second value 2.5 declared");
    }
}

} // namespace

int main()
{
    sa::Runtime::init(true);
    std::printf("device_convert_validation: host <-> device through export_array / import_array\n");
    try
    {
        run(device_convert::DeviceVariant::zero_any);
        run(device_convert::DeviceVariant::zero);
        run(device_convert::DeviceVariant::zero_or_value);
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
