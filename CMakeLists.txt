# NetWarp on Caffe

cmake_minimum_required(VERSION 3.1)
project(NetWarp C CXX)

cmake_policy(SET CMP0054 NEW)

set(NETWARP_DIRECTORY ${CMAKE_SOURCE_DIR}/caffe)
set(CAFFE_UPSTREAM_NAME "CaffeUpstream")

if("${CAFFE_SRC}" STREQUAL "")

  # CAFFE_SRC is set depending on the PREFIX
  set(CAFFE_CLONE TRUE)

  # Default caffe version
  if("${CAFFE_VERSION}" STREQUAL "")
    set(CAFFE_VERSION 691febcb83d6a3147be8e9583c77aefaac9945f8)
  endif()

  # Default repository
  if("${CAFFE_REPOSITORY}" STREQUAL "")
    set(CAFFE_REPOSITORY "https://github.com/BVLC/caffe.git")
  endif()

  message(STATUS "Configuring the project with a fresh clone of Caffe (${CAFFE_REPOSITORY} @ ${CAFFE_VERSION})")
else()
  set(CAFFE_CLONE FALSE)
  message(STATUS "Using an existing Caffe copy from ${CAFFE_SRC}")
endif()

if(UNIX)
  # some issues with CUDA 7.5 / gcc 5.3 +, apparently solved in new versions
  # of CUDA
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES")
endif()

if(CAFFE_CLONE)
  set(_prefix ${CMAKE_BINARY_DIR}/tmp_caffe_clone)
  set(PROJECT_ADD_OPTS
      PREFIX ${_prefix}
      GIT_REPOSITORY "${CAFFE_REPOSITORY}"
      GIT_TAG ${CAFFE_VERSION}
  )
  set(CAFFE_SRC ${_prefix}/src/${CAFFE_UPSTREAM_NAME})
else()
  set(PROJECT_ADD_OPTS
      SOURCE_DIR "${CAFFE_SRC}")
endif()

set(CMAKE_ARGS_IGNORE
    CAFFE_SRC
    CAFFE_REPOSITORY
    CAFFE_VERSION
    CUDA_NVCC_FLAGS # already handled
    CMAKE_PROJECT_NAME
    CMAKE_INSTALL_PREFIX # not passing installation options
    ${PROJECT_NAME}_BINARY_DIR
    ${PROJECT_NAME}_SOURCE_DIR
  )
set(CMAKE_ARGS -DCUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS})
set(CMAKE_CACHE_ARGS)

get_cmake_property(CACHE_VARS CACHE_VARIABLES)
foreach(CACHE_VAR ${CACHE_VARS})
  list(FIND CMAKE_ARGS_IGNORE "${CACHE_VAR}" _index)
  if("${_index}" GREATER -1)
    message(STATUS "Ignoring variable '${CACHE_VAR}'")
    continue()
  endif()

  get_property(CACHE_VAR_TYPE CACHE ${CACHE_VAR} PROPERTY TYPE)
  if(CACHE_VAR_TYPE STREQUAL "UNINITIALIZED")
    set(CACHE_VAR_TYPE)
  else()
    set(CACHE_VAR_TYPE :${CACHE_VAR_TYPE})
  endif()

  # do not pass internal variables
  if(CACHE_VAR_TYPE STREQUAL ":INTERNAL")
    continue()
  endif()

  # do not pass empty variables
  if("${${CACHE_VAR}}" STREQUAL "")
    #message(STATUS "${CACHE_VAR} not initialized or empty, skipping: '${${CACHE_VAR}}'")
    continue()
  endif()

  # prune -NOTFOUND variables
  string(FIND ${${CACHE_VAR}} "-NOTFOUND" _index_not_found REVERSE)
  if(NOT ${_index_not_found} EQUAL -1)
    string(SUBSTRING ${${CACHE_VAR}} ${_index_not_found} -1 _index_not_found_substr)
    if("${_index_not_found_substr}" STREQUAL "-NOTFOUND")
      message(STATUS "${CACHE_VAR} value ends with -NOTFOUND (${${CACHE_VAR}}): skipping")
      continue()
    endif()
  endif()

  if(NOT (DEFINED ${CACHE_VAR}))
    message(STATUS "${CACHE_VAR} is NOT DEFINED: '${${CACHE_VAR}}'")
    continue()
  endif()

  message(STATUS "Passing option ${CACHE_VAR} to ExternalProject_Add")
  string(FIND ${CACHE_VAR} "CMAKE_" _index_cmake)
  if(NOT ${_index_cmake} EQUAL 0)
    set(CMAKE_ARGS ${CMAKE_ARGS} -D${CACHE_VAR}${CACHE_VAR_TYPE}=${${CACHE_VAR}})
  else()
    set(CMAKE_CACHE_ARGS ${CMAKE_CACHE_ARGS} -D${CACHE_VAR}${CACHE_VAR_TYPE}=${${CACHE_VAR}})
  endif()
endforeach()

message(STATUS "CMAKE_ARGS: ${CMAKE_ARGS} / ")
message(STATUS "CMAKE_CACHE_ARGS: ${CMAKE_CACHE_ARGS} / ")



include(ExternalProject)
ExternalProject_Add(
  ${CAFFE_UPSTREAM_NAME}
  ${PROJECT_ADD_OPTS}
  BINARY_DIR "bin"
  UPDATE_COMMAND ""
  PATCH_COMMAND ${CMAKE_COMMAND} -DNETWARP_SOURCE_DIR=${NETWARP_DIRECTORY} -DCAFFE_SOURCE_DIR=${CAFFE_SRC} -P ${NETWARP_DIRECTORY}/cmake/patch_caffe_with_netwarp.cmake
  CMAKE_ARGS ${CMAKE_ARGS}
  CMAKE_CACHE_ARGS ${CMAKE_CACHE_ARGS}
  LOG_DOWNLOAD 1
  LOG_CONFIGURE 1
  LOG_INSTALL 1
  )

add_custom_target(
  runtest
  COMMAND ${CMAKE_COMMAND} --build bin --target runtest
  DEPENDS CaffeUpstream
)

message(STATUS "Optical flow based on dense inverse search included for compilation")
add_subdirectory("external/OF_DIS")
#add_subdirectory(${CAFFE_DIRECTORY} caffe_tmp)
