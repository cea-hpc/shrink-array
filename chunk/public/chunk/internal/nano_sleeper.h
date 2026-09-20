
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

#include <chrono>
#include <random>

namespace shrink_array
{

/// @brief Random sub-microsecond busy-wait, for back-off in spin loops.
class NanoSleeper
{
  private:
    std::mt19937                       gen;  // Random number generator
    std::uniform_int_distribution<int> dist; // Random number generator

  public:
    NanoSleeper() : gen(std::random_device{}()), dist(100, 10000) {} //  One-time initialization

    /// @brief Busy-wait a random 100 ns to 10 us (a back-off that does not
    ///        yield the CPU).
    void sleepRandomNano()
    {
        int delay_ns = dist(gen); // Generate a random delay

        auto start = std::chrono::high_resolution_clock::now();
        while (std::chrono::high_resolution_clock::now() - start < std::chrono::nanoseconds(delay_ns))
        {
            // Busy-wait loop
        }
    }
};

} // namespace shrink_array