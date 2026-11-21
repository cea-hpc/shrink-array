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

// Configure-time probe (cmake/shrink_ompt_probe.cmake): does the OpenMP
// runtime this build links load an OMPT tool that the EXECUTABLE defines?
//
// That is how the library ships its tool (chunk/private/manage_ompt.cpp, linked
// with -rdynamic): the runtime looks ompt_start_tool up in the process. Some
// runtimes never see it -- Ubuntu 24.04's libomp5-18 binds its own weak
// ompt_start_tool at link time (-Bsymbolic-functions) and only searches the
// objects loaded after it (dlsym(RTLD_NEXT)). The library is then built
// without the tool, as with GCC, instead of running with its phase boundaries
// silently missing.
//
// Exit status 0: the runtime called ompt_initialize. Anything else: it did not.

#include <omp-tools.h>
#include <omp.h>

namespace
{
int g_initialized = 0;

int probe_initialize(ompt_function_lookup_t, int, ompt_data_t*)
{
    g_initialized = 1;
    return 1;
}

void probe_finalize(ompt_data_t*) {}
} // namespace

// extern "C" and exported: some omp-tools.h (LLVM 18) do not declare it.
extern "C" __attribute__((visibility("default"))) ompt_start_tool_result_t* ompt_start_tool(unsigned int, const char*)
{
    static ompt_start_tool_result_t result = {&probe_initialize, &probe_finalize, {0}};
    return &result;
}

int main()
{
    // The runtime initialises (and looks for its tool) on the first OpenMP call.
    int threads = omp_get_max_threads();
#pragma omp parallel num_threads(2)
    {
#pragma omp atomic
        threads += 0;
    }
    return g_initialized == 1 && threads > 0 ? 0 : 1;
}
