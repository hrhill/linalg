
set(TESTS norms
          linear_algebra
          factorisation
          operations
          special_types)

set(LINK_LIBS ${BLAS_LIBRARIES}
              ${LAPACK_LIBRARIES}
              ${Boost_TEST_EXEC_MONITOR_LIBRARY})

foreach(test ${TESTS})
    add_executable(${test} ${test}.cpp)
    target_link_libraries(${test} ${LINK_LIBS})
    add_test(${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
endforeach(test)
