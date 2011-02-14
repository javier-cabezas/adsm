set(opencl_test_SRC
    Accelerator.h Accelerator.cpp
    Mode.h Mode.cpp
)

add_gmac_sources(gmac-dbc ${opencl_test_SRC})
