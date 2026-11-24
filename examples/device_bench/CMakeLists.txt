# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
#
# device_bench: device::Array (chunk/device/) against a flat device array, the
# stage-0 measurements of docs/gpu_design.md section 13 on the library's code.
# Built with the device validations (-DCHUNK_DEVICE_CORE_CHECK=ON), in
# CHUNK_DEVICE_LANGUAGE (CUDA or HIP). Only the include path of a library target,
# not its compile options (host flags such as -march would reach the device compiler).

enable_language(${CHUNK_DEVICE_LANGUAGE})
add_executable(device_bench device_bench.cu)
set_source_files_properties(device_bench.cu PROPERTIES LANGUAGE ${CHUNK_DEVICE_LANGUAGE})
set_target_properties(device_bench PROPERTIES ${CHUNK_DEVICE_LANGUAGE}_STANDARD 20
                      ${CHUNK_DEVICE_LANGUAGE}_STANDARD_REQUIRED ON)
target_compile_options(device_bench PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Werror=all-warnings>)
target_include_directories(device_bench PRIVATE
        $<TARGET_PROPERTY:chunk_zero_any,INTERFACE_INCLUDE_DIRECTORIES>
        ${PROJECT_SOURCE_DIR}/fast_allocators/public)
