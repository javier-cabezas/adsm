add_subdirectory(${OS_DIR})
set(trace_SRC
    Tracer.h Tracer-impl.h Tracer.cpp
    Console.h Console.cpp
    Function.h Function.cpp
    Thread.h Thread.cpp)

add_gmac_sources(gmac ${trace_SRC})
