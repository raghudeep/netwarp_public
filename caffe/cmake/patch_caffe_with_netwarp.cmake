# Copyright 2016 Max Planck Society
# Created by Raffi Enficiaud.

# This is an internal helper for the NetWarp code
# that patches the Caffe project

# Copies the files of the netwarp module onto the caffe project


# Gets the source directory
if("${NETWARP_SOURCE_DIR}" STREQUAL "")
  get_filename_component(NETWARP_SOURCE_DIR  "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
endif()
get_filename_component(NETWARP_SOURCE_DIR   "${NETWARP_SOURCE_DIR}" ABSOLUTE)

# Gets the source directory
if("${CAFFE_SOURCE_DIR}" STREQUAL "")
  get_filename_component(current_source_directory   "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  get_filename_component(current_source_directory   "${current_source_directory}/../../caffe_srcx" ABSOLUTE)
  set(CAFFE_SOURCE_DIR ${current_source_directory})
endif()
get_filename_component(CAFFE_SOURCE_DIR   "${CAFFE_SOURCE_DIR}" ABSOLUTE)

message(STATUS "Copying files from ${NETWARP_SOURCE_DIR} to ${CAFFE_SOURCE_DIR}")

file(GLOB_RECURSE netwarp_SRC
     RELATIVE ${NETWARP_SOURCE_DIR}
     "${NETWARP_SOURCE_DIR}/cmake/External/*.*" # patches the glog problem on Ubuntu 16.04
     "${NETWARP_SOURCE_DIR}/utils/*.*"
     "${NETWARP_SOURCE_DIR}/include/*.*"
     "${NETWARP_SOURCE_DIR}/src/*.*"
   )

foreach(_current_file IN LISTS netwarp_SRC)
    # copy
    get_filename_component(current_directory "${CAFFE_SOURCE_DIR}/${_current_file}" DIRECTORY)
    get_filename_component(current_file_without_folder "${_current_file}" NAME)

    #if(("${current_file_without_folder}" STREQUAL "caffe.proto")
    #   AND (EXISTS "${current_directory}/${current_file_without_folder}"))
    #  # removing line ending differences
    #  message(STATUS "${NETWARP_SOURCE_DIR} chk ${_current_file}")
    #  configure_file("${NETWARP_SOURCE_DIR}/${_current_file}" left.proto.tmp NEWLINE_STYLE CRLF)
    #  message(STATUS "${current_directory} chk ${current_file_without_folder}")
    #  configure_file("${current_directory}/${current_file_without_folder}" right.proto.tmp NEWLINE_STYLE CRLF)
    #  message(STATUS "${current_directory} chk ${current_file_without_folder}")

    #  # compares the files, lines ending agnostic
    #  execute_process(
    #    COMMAND ${CMAKE_COMMAND} -E compare_files ./left.proto.tmp ./right.proto.tmp
    #    RESULT_VARIABLE res_different
    #    OUTPUT_VARIABLE out_different
    #  )

    #  # remove the temporary files
    #  file(REMOVE ./left.proto.tmp ./right.proto.tmp)

    #  if(NOT ("${res_different}" STREQUAL "0"))
    #    message(STATUS "[NetWarp] backing up previous 'caffe.proto' to 'caffe.proto.bak'")
    #    configure_file("${current_directory}/${current_file_without_folder}"
    #                   "${current_directory}/${current_file_without_folder}.bak"
    #                   COPYONLY)
    #  endif()

    #endif()

    message(STATUS "[NetWarp] copying file ${current_file_without_folder} to ${current_directory}")
    file(COPY ${NETWARP_SOURCE_DIR}/${_current_file}
         DESTINATION ${current_directory}
       )
endforeach()
