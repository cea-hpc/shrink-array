
/*
 * Copyright 2024-2025 CEA (www.cea.fr)
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

#include <chrono> // std::chrono
#include <random> // std::mt19937, std::uniform_int_distribution

namespace shrink_array
{

/// @brief Random busy-wait of 100 ns to 10 us, for back-off in spin loops.
class NanoSleeper
{
  private:
    std::mt19937                       gen;  // random number generator
    std::uniform_int_distribution<int> dist; // the delay, in nanoseconds

  public:
    /// @brief Seed the generator from std::random_device; delays are drawn in [100, 10000] ns.
    NanoSleeper() : gen(std::random_device{}()), dist(100, 10000) {}

    /// @brief Busy-wait a random 100 ns to 10 us (a back-off that does not
    ///        yield the CPU).
    void sleepRandomNano()
    {
        const int delay_ns = dist(gen);

        const auto start = std::chrono::high_resolution_clock::now();
        while (std::chrono::high_resolution_clock::now() - start < std::chrono::nanoseconds(delay_ns))
        {
            // Busy-wait loop
        }
    }
};

} // namespace shrink_array