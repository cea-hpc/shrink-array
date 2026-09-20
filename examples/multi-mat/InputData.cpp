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

#include <iostream>
#include <string>
#include <random>

#include "Simu.h"

void Simu::inputData()
{
#ifdef INPUT_DATA
    std::string data; // one line of stdin, read under each prompt
    std::cout << "Number of material (default " << nbmat << " ) :";
    std::getline(std::cin, data);

    // Checks if the user entered data or pressed Enter (carriage return)
    if (data.empty())
    {
        std::cout << "default : " << nbmat << " materials\n";
    }
    else
    {
        try
        {
            nbmat = std::stoi(data); // Convert string to integer
            std::cout << "You give : " << nbmat << " materials\n";
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
        }
    }

    int input_mesh_mode = 1;

    std::cout << "Input mesh mode (1 by million 2 real) default 1 : ";

    std::getline(std::cin, data);

    if (data.empty())
    {
        std::cout << "default : " << 1 << "\n";
    }
    else
    {
        try
        {
            input_mesh_mode = std::stoi(data); // Convert string to integer
            std::cout << "You give : " << input_mesh_mode << " input mesh mode\n";
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
        }
    }

    if (input_mesh_mode == 2)
        std::cout << "Mesh size (default 10000000) : ";
    else
        std::cout << "Mesh size (default 10 Millions) : ";

    std::getline(std::cin, data);

    if (data.empty())
    {
        std::cout << "default : " << mesh_size << " mesh size\n";
    }
    else
    {
        try
        {
            mesh_size = std::stoi(data); // Convert string to integer
            if (input_mesh_mode == 1)
                mesh_size *= 1000000;

            std::cout << "You give : " << mesh_size << " mesh size\n";
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
        }
    }

    std::cout << "verbose (default 1) : ";

    std::getline(std::cin, data);

    if (data.empty())
    {
        std::cout << "default : " << true << "\n";
    }
    else
    {
        try
        {
            verbose = std::stoi(data); // Convert string to integer
            if (verbose)
                std::cout << "verbose : true\n";
            else
                std::cout << "verbose : false\n";
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
        }
    }

    std::cout << "Number of iterations (default " << iter_max << " ) : ";

    std::getline(std::cin, data);

    if (data.empty())
    {
        std::cout << "default : " << iter_max << " number of iterations\n";
    }
    else
    {
        try
        {
            iter_max = std::stoi(data); // Convert string to integer
            std::cout << "You give : " << iter_max << " number of iterations\n";
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
        }
    };


    std::cout << "Option : \n";
    std::cout << "   all chunk                        : 1\n";
    std::cout << "   all vector                       : 2\n";
    std::cout << "   random vector / chunk (default)  : 3\n";
    std::cout << "   random vector / chunk with rate  : 4\n";

    std::getline(std::cin, data);

    if (data.empty())
    {
        std::cout << "default : " << option_chunk << "\n";
    }
    else
    {
        try
        {
            option_chunk = std::stoi(data); // Convert string to integer
            std::cout << "You give : " << option_chunk << "\n\n";

            if (option_chunk < 1 || option_chunk > 4)
            {
                throw std::out_of_range("The value_compute is not in the specified range.");
            }
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
        }

        if (option_chunk == 4)
        {
            std::cout << "Pourcentage of chunk array : \n";
            std::getline(std::cin, data);

            if (data.empty())
            {
                std::cout << "default : " << 90 << "\n";
            }
            else
            {
                try
                {
                    chunk_rate = std::stoi(data); // Convert string to integer
                    std::cout << "You give : " << chunk_rate << "\n\n";
                }
                catch (const std::invalid_argument& e)
                {
                    std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
                }
            }
        }
    }

#endif
    dis.resize(nbmat);

    switch (getOptionChunk())
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
        {
            // Initialize the random number generator

            std::random_device rd0;

            std::mt19937                    gen(rd0());
            std::uniform_int_distribution<> distrib(0, 1);

            int nb_chunk = 0;

            // Loop to distribute true or false values with a 50/50 chance
            for (int m = 0; m < nbmat; ++m)
            {
                dis[m] = distrib(gen) != 0; // 0 corresponds to false, 1 corresponds to true
                nb_chunk += dis[m];
                // Shows the distributed value_compute
                std::cout << "Material #" << m << ": " << (dis[m] ? "chunk" : "vector") << "\n";
            }

            std::cout << "\nNumber of chunk array : "  << nb_chunk << " on " << nbmat << " materials \n\n";

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

        case 4:
        {
            // Initialize the random number generator

            std::random_device rd0;

            std::mt19937                    gen(rd0());
            std::uniform_int_distribution<> distrib(0, 100);

            int nb_chunk = 0;

            // Loop to distribute true or false values with a 50/50 chance
            for (int m = 0; m < nbmat; ++m)
            {
                dis[m] = distrib(gen) < chunk_rate;
                nb_chunk += dis[m];
                // Shows the distributed value_compute
                std::cout << "material #" << m << ": " << (dis[m] ? "chunk" : "vector") << "\n";
            }
            std::cout << "\nNumber of chunk array : "  << nb_chunk << " on " << nbmat << " materials \n\n";

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

#ifdef INPUT_DATA
    std::string data;
    if (option_chunk != 2)
    {
        std::cout << "Compression frequency ( default " << freq_compress_default << " ) : ";

        std::getline(std::cin, data);

        if (data.empty())
        {
            std::cout << "default : " << freq_compress << " compression frequency\n";
        }
        else
        {
            try
            {
                freq_compress = std::stoi(data); // Convert string to integer
                std::cout << "You give : " << freq_compress << " compression frequency\n";
            }
            catch (const std::invalid_argument& e)
            {
                std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
            }
        }
    }

    std::cout << "reset to zero frequency (default " << freq_rtz_default << " ) : ";

    std::getline(std::cin, data);

    if (data.empty())
    {
        freq_rtz = freq_rtz_default;
        std::cout << "default : " << freq_rtz << " reset to zero frequency\n";
    }
    else
    {
        try
        {
            freq_rtz = std::stoi(data); // Convert string to integer
            std::cout << "You give : " << freq_rtz << " reset to zero frequency\n";
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
        }
    }

    constexpr double eps = 1e-10;

    std::cout << "\n";

    if (!all_vector)
    {
        std::cout << "Average percentage occupation for chunk material ( default " << pc_chunk << " ) : ";

        std::getline(std::cin, data);

        if (data.empty())
        {
            std::cout << "default : " << pc_chunk << "%\n";
        }
        else
        {
            try
            {
                pc_chunk = std::stod(data); // Convert string to integer
                std::cout << "You give : " << pc_chunk << "%\n";
                if (pc_chunk < eps || pc_chunk > 100 - eps)
                {
                    throw std::out_of_range("The value_compute is not in the specified range.");
                }
            }
            catch (const std::invalid_argument& e)
            {
                std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
            }
        }
    }

    if (!all_chunk)
    {
        std::cout << "Average percentage occupation for vector material ( default " << pc_vector << " ) : ";

        std::getline(std::cin, data);

        if (data.empty())
        {
            std::cout << "default : " << pc_vector << "%\n";
        }
        else
        {
            try
            {
                pc_vector = std::stod(data); // Convert string to integer
                std::cout << "You give : " << pc_vector << "%\n";
                if (pc_vector < eps || pc_vector > 100 - eps)
                {
                    throw std::out_of_range("The value_compute is not in the specified range.");
                }
            }
            catch (const std::invalid_argument& e)
            {
                std::cerr << "Error: Conversion failed. The string is not a valid number" << std::endl;
            }
        }
    }
#else
    std::cout << nbmat << " materials\n";
    std::cout << mesh_size << " mesh size\n";
    std::cout << iter_max << " number of iterations\n";
    std::cout << option_chunk << " option chunk\n";
    std::cout << pc_chunk << "% fill rate chunk\n";
    std::cout << pc_vector << "% fill rate vector\n";
    std::cout << freq_rtz << " frequency reset to zero frequency\n";
    std::cout << freq_compress << " frequency compression frequency\n";
#endif
}