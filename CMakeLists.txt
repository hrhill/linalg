cmake_minimum_required(VERSION 2.6)
project(LINALG)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/modules/")

include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)
include(CppCheck)
include(ClangFormat)

find_package(Boost 1.50 COMPONENTS test_exec_monitor REQUIRED)
find_package(Doxygen)
find_package(BLAS)
find_package(LAPACK)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic -Wfatal-errors -std=c++1y")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${Boost_INCLUDE_DIR}")
link_directories("${Boost_LIBRARY_DIRS}")

add_subdirectory(test)
add_subdirectory(example)

#------------------------------------ Documentation setup ---------------------
if(DOXYGEN_FOUND)
configure_file(${PROJECT_SOURCE_DIR}/Doxyfile.in
               ${PROJECT_BINARY_DIR}/Doxyfile)
add_custom_target(doc ${DOXYGEN_EXECUTABLE}
    ${PROJECT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

#------------------------------------ Installation details --------------------
set(HEADER_INST_DIR "${CMAKE_INSTALL_PREFIX}/include")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION "${HEADER_INST_DIR}"
        FILES_MATCHING PATTERN "*.hpp")
