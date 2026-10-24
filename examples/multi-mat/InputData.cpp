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

/**
 * @file InputData.cpp
 * @brief Simu::input_data(): the parameters of the run and the storage of
 *        every material.
 *
 * Interactive under multi_mat::interactive_input (one prompt per
 * parameter, Enter keeps the default); the defaults of Simu.h otherwise,
 * printed. The two paths are one function: the prompts are ordinary code
 * behind `if constexpr`, compiled in every build.
 */

#include <iostream>    // std::cout, std::cerr, std::cin
#include <random>      // std::mt19937, std::uniform_int_distribution
#include <string>      // std::string, std::stoi, std::stod
#include <stdexcept>   // std::invalid_argument, std::out_of_range
#include <algorithm>   // std::max
#include <format>      // std::format
#include <type_traits> // std::is_floating_point_v (shown)
#include <print>       // std::println, std::print

#include "config.h" // multi_mat::interactive_input
#include "Simu.h"   // Simu::input_data

namespace
{

/**
 * @brief One prompt on stdin: Enter keeps @p value, anything else is parsed
 *        by @p parse and, when @p valid accepts it, stored.
 * @param prompt  What is asked.
 * @param name    What the value is, for the echo.
 * @param value   The parameter, holding its default on entry.
 * @param parse   std::stoi / std::stod, or a lambda.
 * @param valid   Range check on the parsed value; a refusal is reported and
 *                the default kept.
 */
/// @brief @p v as a default ostream showed it: {:g} for a floating-point one.
template <typename T>
std::string shown(const T& v)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return std::format("{:g}", v);
    }
    else
    {
        return std::format("{}", v);
    }
}

template <typename T, typename Parse, typename Valid>
void ask(const char* prompt, const char* name, T& value, Parse parse, Valid valid)
{
    std::print("{} (default {}) : ", prompt, shown(value));

    std::string data;
    std::getline(std::cin, data);

    if (data.empty())
    {
        std::println("default : {} {}", shown(value), name);
        return;
    }
    try
    {
        const T parsed = parse(data);
        if (!valid(parsed))
        {
            throw std::out_of_range("The value is not in the specified range.");
        }
        value = parsed;
        std::println("You give : {} {}", shown(value), name);
    }
    catch (const std::invalid_argument&)
    {
        std::println(stderr, "Error: Conversion failed. The string is not a valid number");
    }
    catch (const std::out_of_range& e)
    {
        std::println(stderr, "Error: {} default kept: {}", e.what(), shown(value));
    }
}

/// @brief ask() for an int, any value.
[[maybe_unused]] void ask_int(const char* prompt, const char* name, int& value)
{
    ask(prompt, name, value, [](const std::string& text) { return std::stoi(text); }, [](int) { return true; });
}

/// @brief ask() for a percentage, in ]0, 100[.
[[maybe_unused]] void ask_percentage(const char* prompt, const char* name, double& value)
{
    constexpr double eps = 1e-10;
    ask(
        prompt, name, value, [](const std::string& text) { return std::stod(text); },
        [](double value) { return value >= eps && value <= 100 - eps; });
}

} // namespace

void Simu::input_data()
{
    if constexpr (multi_mat::interactive_input)
    {
        ask_int("Number of material", "materials", nbmat);

        int input_mesh_mode = 1;
        ask_int("Input mesh mode (1 by million 2 real)", "input mesh mode", input_mesh_mode);
        if (input_mesh_mode == 2)
        {
            ask_int("Mesh size", "mesh size", mesh_size);
        }
        else
        {
            int millions = mesh_size / 1000000;
            ask_int("Mesh size, in millions", "million cells", millions);
            mesh_size = millions * 1000000;
        }

        int verbose_int = verbose ? 1 : 0;
        ask_int("verbose", "verbose", verbose_int);
        verbose = verbose_int != 0;

        ask_int("Number of iterations", "number of iterations", iter_max);

        std::println("Option : ");
        std::println("   all chunk                        : 1");
        std::println("   all vector                       : 2");
        std::println("   random vector / chunk (default)  : 3");
        std::println("   random vector / chunk with rate  : 4");
        ask(
            "Option", "option", option_chunk, [](const std::string& text) { return std::stoi(text); },
            [](int v) { return v >= 1 && v <= 4; });

        if (option_chunk == 4)
        {
            ask_int("Pourcentage of chunk array", "% chunk", chunk_rate);
        }
    }

    dis.resize(nbmat);

    switch (get_option_chunk())
    {
        case 1:
        {
            all_chunk = true;
            dis.assign(nbmat, true);
        }
        break;
        case 2:
        {
            all_vector = true;
            dis.assign(nbmat, false);
        }
        break;
        case 3:
        case 4:
        {
            // Option 3: a 50/50 draw; option 4: chunk with probability
            // chunk_rate %.
            std::random_device rd0;

            std::mt19937                    gen(rd0());
            std::uniform_int_distribution<> distrib(0, option_chunk == 3 ? 1 : 100);

            const int threshold = option_chunk == 3 ? 1 : chunk_rate;

            int nb_chunk = 0;

            for (int m = 0; m < nbmat; ++m)
            {
                dis[m] = distrib(gen) < threshold;
                nb_chunk += dis[m];
                std::println("Material #{}: {}", m, dis[m] ? "chunk" : "vector");
            }

            std::println("\nNumber of chunk array : {} on {} materials \n", nb_chunk, nbmat);

            if (nb_chunk == nbmat)
            {
                all_chunk = true;
            }
            else if (nb_chunk == 0)
            {
                all_vector = true;
            }
        }
        break;
        default:
            break; // 1..4, validated when read above
    }

    const int freq_compress_default = std::max(iter_max / 10, 10);

    freq_compress = freq_compress_default;

    freq_rtz = freq_rtz_default;

    if constexpr (multi_mat::interactive_input)
    {
        if (option_chunk != 2)
        {
            ask_int("Compression frequency", "compression frequency", freq_compress);
        }

        ask_int("reset to zero frequency", "reset to zero frequency", freq_rtz);

        std::println("");

        if (!all_vector)
        {
            ask_percentage("Average percentage occupation for chunk material", "%", pc_chunk);
        }

        if (!all_chunk)
        {
            ask_percentage("Average percentage occupation for vector material", "%", pc_vector);
        }
    }
    else
    {
        std::println("{} materials", nbmat);
        std::println("{} mesh size", mesh_size);
        std::println("{} number of iterations", iter_max);
        std::println("{} option chunk", option_chunk);
        std::println("{:g}% fill rate chunk", pc_chunk);
        std::println("{:g}% fill rate vector", pc_vector);
        std::println("{} frequency reset to zero frequency", freq_rtz);
        std::println("{} frequency compression frequency", freq_compress);
    }
}
