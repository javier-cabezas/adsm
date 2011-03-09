set(core_test_SRC
    Context.h Context.cpp
    Mode.h Mode.cpp
    Accelerator.h Accelerator.cpp  
    IOBuffer.h IOBuffer.cpp
    Process.h Process.cpp
#    Kernel.h Kernel.cpp
)

add_gmac_sources(gmac-dbc ${core_test_SRC})
