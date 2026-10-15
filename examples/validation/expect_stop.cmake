# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
# A test that passes when a program STOPS with a given message: ctest counts
# an abort as a failure whatever PASS_REGULAR_EXPRESSION says.
#   cmake -DEXE=<program> -DARGS=<arg;...> -DEXPECT=<regex> -P expect_stop.cmake
execute_process(COMMAND ${EXE} ${ARGS} RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
if (_rc EQUAL 0)
  message(FATAL_ERROR "${EXE} ${ARGS}: exited normally, expected it to stop\n${_out}")
endif()
if (NOT _out MATCHES "${EXPECT}")
  message(FATAL_ERROR "${EXE} ${ARGS}: stopped (${_rc}) without the expected message \"${EXPECT}\"\n${_out}")
endif()
message(STATUS "${EXE} ${ARGS}: stopped as expected (${_rc})")
