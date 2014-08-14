
cmake_minimum_required(VERSION 2.6)
project(LINEAR_ALGEBRA)

include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

#set(CMAKE_CXX_COMPILER "clang++")
set(BUILD_SHARED_LIBS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LINEAR_ALGEBRA_SOURCE_DIR}/modules/")
set(CMAKE_CXX_FLAGS  "-Wall -pedantic -Wfatal-errors -std=c++1y")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

find_package(Boost 1.50 COMPONENTS test_exec_monitor program_options REQUIRED)
find_package(Doxygen)
find_package(BLAS)
find_package(LAPACK)
find_package(UBlas)
find_package(MTL)
find_package(Blaze)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory(test)
add_subdirectory(example)
add_subdirectory(benchmark)

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
