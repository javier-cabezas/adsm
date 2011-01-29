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
    add_gmac_test_include(${CMAKE_CURRENT_SOURCE_DIR}/${gtest_DIR}/include) 
    add_gmac_test_library(gtest)

    if(MAKE_TESTS_ATF)
        # Add ATF Testing Framework
        include(ExternalProject)

        ExternalProject_Add(
            ATF-0-12
            URL ftp://ftp.netbsd.org/pub/NetBSD/misc/jmmv/atf/0.12/atf-0.12.tar.gz
            SOURCE_DIR atf-0.12
            BINARY_DIR atf-0.12-build
            CONFIGURE_COMMAND ../atf-0.12/configure --prefix=${CMAKE_INSTALL_PREFIX}/local
            BUILD_COMMAND make -j 4
            )
    endif(MAKE_TESTS_ATF)
endif(MAKE_TESTS)
