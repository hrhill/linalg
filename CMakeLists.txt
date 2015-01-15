
cmake_minimum_required(VERSION 2.6)
project(LINEAR_ALGEBRA)

include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)
include(ExternalProject)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LINEAR_ALGEBRA_SOURCE_DIR}/build/")
set(CMAKE_CXX_FLAGS  "-Wall -pedantic -Wfatal-errors -std=c++1y")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

find_package(Boost 1.50 COMPONENTS test_exec_monitor REQUIRED)
find_package(Doxygen)
find_package(BLAS)
find_package(LAPACK)
find_package(UBlas)
find_package(MTL)
find_package(Blaze)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${Boost_INCLUDE_DIR}")
link_directories("${Boost_LIBRARY_DIR}")

add_subdirectory(test)
add_subdirectory(example)
add_subdirectory(benchmark)

ExternalProject_Add(numeric_bindings
    GIT_REPOSITORY "https://github.com/hrhill/numeric_bindings.git"
    PREFIX "deps/numeric_bindings"
    BINARY_DIR ${PROJ_BINARY_DIR}
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${PROJECT_BINARY_DIR}
)

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
