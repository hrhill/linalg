set(PROG gemm)

foreach(prog ${PROGS})
    add_executable(bench_${prog} ${prog}.cpp)
    target_link_libraries(bench_${prog} ${BLAS_LIBRARIES}
                                        ${LAPACK_LIBRARIES})
endforeach(prog)
