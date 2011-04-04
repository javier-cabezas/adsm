set(core_test_SRC
    Accelerator.h Accelerator.cpp  
    IOBuffer.h IOBuffer.cpp)

add_gmac_sources(gmac-dbc ${core_test_SRC})
