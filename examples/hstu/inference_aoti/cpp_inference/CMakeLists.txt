cmake_minimum_required(VERSION 3.18)
project(inference_hstu_gr_ranking_exported_model LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{Torch_DIR}")

set(NVE_INSTALL_ROOT "/opt/nve"
    CACHE PATH "Root containing versioned NVE runtime installations")
set(NVE_2605_ROOT "/workspace/deps/nve-26.05"
    CACHE PATH "NVE 26.05 source tree")
set(NVE_2606_ROOT "/workspace/deps/nve-26.06"
    CACHE PATH "Optional NVE 26.06 source tree")
set(NVE_2607_ROOT "/workspace/deps/nve-26.07"
    CACHE PATH "NVE 26.07 source tree")
set(NVE_2605_LIB_DIR "${NVE_INSTALL_ROOT}/26.05/python/pynve"
    CACHE PATH "NVE 26.05 native-library directory")
set(NVE_2606_LIB_DIR "${NVE_INSTALL_ROOT}/26.06/python/pynve"
    CACHE PATH "Optional NVE 26.06 native-library directory")
set(NVE_2607_LIB_DIR "${NVE_INSTALL_ROOT}/26.07/python/pynve"
    CACHE PATH "NVE 26.07 native-library directory")
set(NVE_2605_REPLAY_DIR "${NVE_INSTALL_ROOT}/26.05/replay"
    CACHE PATH "NVE 26.05 replay-plugin directory")
set(NVE_2606_REPLAY_DIR "${NVE_INSTALL_ROOT}/26.06/replay"
    CACHE PATH "NVE 26.06 replay-plugin directory")
set(NVE_2607_REPLAY_DIR "${NVE_INSTALL_ROOT}/26.07/replay"
    CACHE PATH "NVE 26.07 replay-plugin directory")
set(NLOHMANN_JSON_ROOT "${NVE_2607_ROOT}/third_party/json"
    CACHE PATH "nlohmann/json checkout bundled with NVE 26.07")
set(NVTX_INCLUDE_DIR "/workspace/deps/NVTX/c/include"
    CACHE PATH "Pinned NVTX C/C++ include directory")

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Every target in this build must use the same libstdc++ ABI as the PyTorch
# package, including the NVE-free manager whose C++ interface uses std::string.
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_compile_options(${TORCH_CXX_FLAGS_LIST})

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(
  "${NLOHMANN_JSON_ROOT}"
  "${CMAKE_CURRENT_BINARY_DIR}/nlohmann-json"
  EXCLUDE_FROM_ALL
)

find_library(FBGEMM_GPU_LIB fbgemm_gpu_py
  PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
find_library(FBGEMM_HSTU_EXPERIMENTAL_LIB fbgemm_gpu_experimental_hstu
  PATHS /usr/local/lib/python3.12/dist-packages/hstu ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)

find_library(NVE_2605_TORCH_OPS_LIB nve-torch-ops
  PATHS "${NVE_2605_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(NVE_2605_COMMON_LIB nve-common
  PATHS "${NVE_2605_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
if(IS_DIRECTORY "${NVE_2606_ROOT}" AND IS_DIRECTORY "${NVE_2606_LIB_DIR}")
  find_library(NVE_2606_TORCH_OPS_LIB nve-torch-ops
    PATHS "${NVE_2606_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  find_library(NVE_2606_COMMON_LIB nve-common
    PATHS "${NVE_2606_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  set(NVE_2606_AVAILABLE ON)
else()
  set(NVE_2606_AVAILABLE OFF)
  message(STATUS
    "NVE 26.06 source or native libraries not found; skipping its replay plugin")
endif()
find_library(NVE_2607_TORCH_OPS_LIB nve-torch-ops
  PATHS "${NVE_2607_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(NVE_2607_COMMON_LIB nve-common
  PATHS "${NVE_2607_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

# These shared libraries contain only the HSTU/KV custom operators used by
# both version-independent replay executables.
add_library(hstu_cuda_ops_runtime SHARED
  ../../../commons/ops/cuda_ops/csrc/kjt_aux_op.cpp
  ../../../commons/ops/cuda_ops/csrc/jagged_tensor_op_cuda.cpp
  ../../../commons/ops/cuda_ops/csrc/jagged_tensor_op_kernel.cu
)
target_link_libraries(hstu_cuda_ops_runtime PRIVATE ${TORCH_LIBRARIES})
target_include_directories(hstu_cuda_ops_runtime PRIVATE
  ${TORCH_INCLUDE_DIRS}
  "${NVTX_INCLUDE_DIR}"
)
target_compile_features(hstu_cuda_ops_runtime PRIVATE cxx_std_17)
set_target_properties(hstu_cuda_ops_runtime PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

add_library(paged_kvcache_ops_runtime SHARED
  ../../../commons/ops/cuda_ops/csrc/paged_kvcache_ops_cuda.cpp
  ../../../commons/ops/cuda_ops/csrc/paged_kvcache_ops_kernel.cu
)
target_link_libraries(paged_kvcache_ops_runtime PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
  /workspace/deps/nvcomp/lib/libnvcomp_static.a
)
target_include_directories(paged_kvcache_ops_runtime PRIVATE
  ${TORCH_INCLUDE_DIRS}
  /workspace/deps/nvcomp/include
  ../../../commons/ops/cuda_ops/csrc
)
target_compile_features(paged_kvcache_ops_runtime PRIVATE cxx_std_20)
set_target_properties(paged_kvcache_ops_runtime PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# The manager parses metadata and dlopens one fixed plugin path. It has no NVE
# include path or link dependency.
add_library(recsys_nve_loader_manager STATIC nve_loader_plugin.cpp)
target_include_directories(recsys_nve_loader_manager PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(recsys_nve_loader_manager
  PRIVATE nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS}
)
target_compile_definitions(recsys_nve_loader_manager PRIVATE
  RECSYS_NVE_2605_PLUGIN_PATH="${NVE_2605_REPLAY_DIR}/librecsys_nve_loader.so"
  RECSYS_NVE_2606_PLUGIN_PATH="${NVE_2606_REPLAY_DIR}/librecsys_nve_loader.so"
  RECSYS_NVE_2607_PLUGIN_PATH="${NVE_2607_REPLAY_DIR}/librecsys_nve_loader.so"
)
target_compile_features(recsys_nve_loader_manager PUBLIC cxx_std_17)
set_target_properties(recsys_nve_loader_manager PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(NOT NVE_CACHE_LINE_SIZE)
  if(WIN32)
    set(NVE_CACHE_LINE_SIZE "64")
  else()
    execute_process(
      COMMAND getconf LEVEL1_DCACHE_LINESIZE
      OUTPUT_VARIABLE NVE_CACHE_LINE_SIZE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
endif()

function(add_versioned_nve_loader_plugin
    target generation nve_root nve_lib_dir replay_dir torch_ops_lib common_lib)
  add_library(${target} MODULE nve_loader_plugin_impl.cu)
  set(version_script "${CMAKE_CURRENT_BINARY_DIR}/${target}.map")
  file(GENERATE OUTPUT "${version_script}" CONTENT
    "{\n  global: recsys_nve_loader_get_api_v1;\n  local: *;\n};\n")
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${nve_root}"
    "${nve_root}/include"
    "${nve_root}/third_party/json/include"
    "${NVTX_INCLUDE_DIR}"
    ${TORCH_INCLUDE_DIRS}
  )
  target_compile_definitions(${target} PRIVATE
    RECSYS_NVE_GENERATION=${generation}
    RECSYS_NVE_LIBRARY_DIR="${nve_lib_dir}"
    NVE_FEATURE_HT_PART_FNV1A=1
    NVE_FEATURE_HT_PART_MURMUR=1
    NVE_FEATURE_HT_PART_RRXMRRXMSX0=1
    NVE_FEATURE_HT_PART_STD_HASH=1
    NVE_FEATURE_HT_MASK_64=1
    NVE_FEATURE_HT_MASK_32=1
    NVE_FEATURE_HT_MASK_16=1
    NVE_FEATURE_HT_MASK_8=1
    NVE_FEATURE_HT_KEY_64=1
    NVE_FEATURE_HT_KEY_32=1
    NVE_FEATURE_HT_KEY_16=1
    NVE_FEATURE_HT_KEY_8=1
    NVE_CACHE_LINE_SIZE=${NVE_CACHE_LINE_SIZE}
  )
  target_link_options(${target} PRIVATE
    "LINKER:--no-as-needed"
    "LINKER:--version-script=${version_script}"
  )
  target_link_libraries(${target} PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    ${TORCH_LIBRARIES}
    "${torch_ops_lib}"
    "${common_lib}"
    ${CMAKE_DL_LIBS}
  )
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME recsys_nve_loader
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${target}"
    BUILD_RPATH "${nve_lib_dir}"
    INSTALL_RPATH "\$ORIGIN/../python/pynve"
  )
  install(TARGETS ${target} LIBRARY DESTINATION "${replay_dir}")
endfunction()

add_versioned_nve_loader_plugin(
  recsys_nve_loader_2605 2605
  "${NVE_2605_ROOT}" "${NVE_2605_LIB_DIR}" "${NVE_2605_REPLAY_DIR}"
  "${NVE_2605_TORCH_OPS_LIB}" "${NVE_2605_COMMON_LIB}"
)
set(RECSYS_NVE_LOADER_TARGETS recsys_nve_loader_2605)
if(NVE_2606_AVAILABLE)
  add_versioned_nve_loader_plugin(
    recsys_nve_loader_2606 2606
    "${NVE_2606_ROOT}" "${NVE_2606_LIB_DIR}" "${NVE_2606_REPLAY_DIR}"
    "${NVE_2606_TORCH_OPS_LIB}" "${NVE_2606_COMMON_LIB}"
  )
  list(APPEND RECSYS_NVE_LOADER_TARGETS recsys_nve_loader_2606)
endif()
add_versioned_nve_loader_plugin(
  recsys_nve_loader_2607 2607
  "${NVE_2607_ROOT}" "${NVE_2607_LIB_DIR}" "${NVE_2607_REPLAY_DIR}"
  "${NVE_2607_TORCH_OPS_LIB}" "${NVE_2607_COMMON_LIB}"
)
list(APPEND RECSYS_NVE_LOADER_TARGETS recsys_nve_loader_2607)

add_executable(
  inference_hstu_gr_ranking_exported_model
  inference_hstu_gr_ranking_exported_model.cpp
)
add_executable(
  inference_hstu_gr_ranking_kvcache_exported_model
  inference_hstu_gr_ranking_kvcache_exported_model.cpp
)

target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
  ${TORCH_LIBRARIES}
  hstu_cuda_ops_runtime
  recsys_nve_loader_manager
  ${CMAKE_DL_LIBS}
)
target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
  ${TORCH_LIBRARIES}
  hstu_cuda_ops_runtime
  paged_kvcache_ops_runtime
  recsys_nve_loader_manager
  ${CMAKE_DL_LIBS}
)
target_include_directories(inference_hstu_gr_ranking_exported_model PRIVATE
  ${TORCH_INCLUDE_DIRS}
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(
  inference_hstu_gr_ranking_kvcache_exported_model PRIVATE
  ${TORCH_INCLUDE_DIRS}
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(inference_hstu_gr_ranking_exported_model PRIVATE
  HSTU_CUDA_OPS_AVAILABLE=1)
target_compile_definitions(
  inference_hstu_gr_ranking_kvcache_exported_model PRIVATE
  HSTU_CUDA_OPS_AVAILABLE=1)

if(FBGEMM_GPU_LIB)
  message(STATUS "Found fbgemm_gpu library: ${FBGEMM_GPU_LIB}")
  target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE
    ${FBGEMM_GPU_LIB})
  target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE
    ${FBGEMM_GPU_LIB})
  target_compile_definitions(inference_hstu_gr_ranking_exported_model PRIVATE
    FBGEMM_GPU_AVAILABLE=1)
  target_compile_definitions(
    inference_hstu_gr_ranking_kvcache_exported_model PRIVATE
    FBGEMM_GPU_AVAILABLE=1)
else()
  message(STATUS "fbgemm_gpu library not found; replay will dlopen it")
endif()

if(FBGEMM_HSTU_EXPERIMENTAL_LIB)
  message(STATUS
    "Found fbgemm experimental hstu library: ${FBGEMM_HSTU_EXPERIMENTAL_LIB}")
  target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE
    ${FBGEMM_HSTU_EXPERIMENTAL_LIB})
  target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE
    ${FBGEMM_HSTU_EXPERIMENTAL_LIB})
else()
  message(STATUS "fbgemm experimental hstu library will be dlopened")
endif()

target_compile_features(inference_hstu_gr_ranking_exported_model PRIVATE cxx_std_17)
target_compile_options(inference_hstu_gr_ranking_exported_model PRIVATE -w)
target_compile_features(
  inference_hstu_gr_ranking_kvcache_exported_model PRIVATE cxx_std_17)
target_compile_options(
  inference_hstu_gr_ranking_kvcache_exported_model PRIVATE -w)

add_dependencies(
  inference_hstu_gr_ranking_exported_model
  ${RECSYS_NVE_LOADER_TARGETS}
)
add_dependencies(
  inference_hstu_gr_ranking_kvcache_exported_model
  ${RECSYS_NVE_LOADER_TARGETS}
)
