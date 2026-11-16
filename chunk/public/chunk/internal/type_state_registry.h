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

#pragma once

/**
 * @file type_state_registry.h
 * @brief Type-erased registry of every per-type chunk manager of the process.
 *
 * @details The declarations (type_state_registry_decl.h) and the template bodies
 *          (type_state_registry_inline.h), kept apart so that a header can name
 *          TypeStateRegistry without pulling in the full zero/zero_one_value/zero_any
 *          definitions.
 *
 * @see type_state.h for the TypeStateInterface interface this class dispatches
 *      through, and phases.h for the phase transitions that drive it.
 */

#include "chunk/internal/type_state_registry_decl.h"   // TypeStateRegistry, the declarations
#include "chunk/internal/type_state_registry_inline.h" // the template bodies
