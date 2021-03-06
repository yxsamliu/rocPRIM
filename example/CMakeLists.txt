# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function(meta_add_rocprim_example EXAMPLE_SOURCE ROCPRIM_TARGET)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})
  target_link_libraries(${EXAMPLE_TARGET}
    PRIVATE
      ${ROCPRIM_TARGET}
  )
  foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
        --amdgpu-target=${amdgpu_target}
    )
  endforeach()
  set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
  )
endfunction()

function(add_rocprim_example_hc EXAMPLE_SOURCE)
  meta_add_rocprim_example(${EXAMPLE_SOURCE} rocprim_hc)
endfunction()

function(add_rocprim_example_hip EXAMPLE_SOURCE)
  meta_add_rocprim_example(${EXAMPLE_SOURCE} rocprim_hip)
endfunction()

# ****************************************************************************
# rocPRIM examples
# ****************************************************************************
 
add_rocprim_example_hip(example_hip_temporary_storage.cpp)
add_rocprim_example_hc(example_hc_temporary_storage.cpp)
