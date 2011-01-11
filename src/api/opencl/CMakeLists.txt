add_subdirectory(memory)

set(api_lib_SRC
    opencl.cpp)

set(api_SRC
    Accelerator.h Accelerator-impl.h Accelerator.cpp Error.cpp
   )

add_gmac_sources(gmac ${api_SRC})
add_gmac_sources(gmac-lib ${api_lib_SRC})
