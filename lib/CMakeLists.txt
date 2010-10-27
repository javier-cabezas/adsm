if(MAKE_TESTS)
    # Add Google Testing Framework Dependency
    set(gtest_DIR gtest-1.5.0)
    set(gtest_SRC
        ${gtest_DIR}/src/gtest-internal-inl.h
        ${gtest_DIR}/src/gtest.cc
        ${gtest_DIR}/src/gtest-death-test.cc
        ${gtest_DIR}/src/gtest-filepath.cc
        ${gtest_DIR}/src/gtest-port.cc
        ${gtest_DIR}/src/gtest-test-part.cc
        ${gtest_DIR}/src/gtest-typed-test.cc)

    include_directories(${gtest_DIR}/include ${gtest_DIR})
    add_library(gtest STATIC ${gtest_SRC})
    add_gmac_test_include(${gtest_DIR}/include) 
    add_gmac_test_library(gtest)
endif(MAKE_TESTS)
