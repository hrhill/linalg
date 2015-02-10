set(PROG gemm)

foreach(prog ${PROGS})
    add_executable(bench_${prog} ${prog}.cpp)
    target_link_libraries(bench_${prog} ${LAPACK_LIBRARIES}
                                        ${BLAS_LIBRARIES})
endforeach(prog)
