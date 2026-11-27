# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
#
# The Arcane facade of device::Array (chunk/device/arcane_adaptor.h): an installed Arcane
# built with CUDA or HIP (find_package(Arcane): Arcane_ROOT or CMAKE_PREFIX_PATH), its
# accelerator macros as its samples use them (arcane/samples_build/samples/
# standalone_accelerator), the CMake target chunk_device (shrink_array::device) for the
# include path. CHUNK_ARCANE_GPU_ARCH: the device architectures of these targets (Arcane's
# configuration forces a default of its own on CMAKE_CUDA_ARCHITECTURES).
set(CHUNK_ARCANE_GPU_ARCH "" CACHE STRING "Device architectures of the Arcane examples: 80, 90 (CUDA), gfx942 (HIP); empty: Arcane's default")
find_package(Arcane REQUIRED)
if (NOT ARCANE_HAS_ACCELERATOR)
  message(FATAL_ERROR "CHUNK_DEVICE_ARCANE: this Arcane (${Arcane_DIR}) has no accelerator support")
endif()
arcane_accelerator_enable()

foreach (_t IN ITEMS arcane_getting_started arcane_validation)
  add_executable(${_t} ${_t}.cc)
  arcane_add_arcane_libraries_to_target(${_t})
  arcane_accelerator_add_source_files(${_t}.cc)
  arcane_accelerator_add_to_target(${_t})
  target_link_libraries(${_t} PRIVATE chunk_device)
  # The device language Arcane compiles the source in: C++20 there too.
  set_target_properties(${_t} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
  if (NOT CHUNK_ARCANE_GPU_ARCH STREQUAL "")
    if (ARCANE_HAS_CUDA)
      set_target_properties(${_t} PROPERTIES CUDA_ARCHITECTURES "${CHUNK_ARCANE_GPU_ARCH}")
    elseif (ARCANE_HAS_HIP)
      set_target_properties(${_t} PROPERTIES HIP_ARCHITECTURES "${CHUNK_ARCANE_GPU_ARCH}")
    endif()
  endif()
endforeach()

# The default queue is a host one unless the runtime is named.
if (ARCANE_HAS_CUDA)
  set(_runtime cuda)
else()
  set(_runtime hip)
endif()
foreach (_t IN ITEMS arcane_getting_started arcane_validation)
  add_test(NAME ${_t} COMMAND ${_t} -A,AcceleratorRuntime=${_runtime})
  set_property(TEST ${_t} PROPERTY LABELS validation)
endforeach()
