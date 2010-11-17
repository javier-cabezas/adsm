set(trace_SRC
    Tracer.h Tracer-impl.h
    Function.h Function.cpp
    Thread.h Thread.cpp)

add_gmac_sources(gmac ${trace_SRC})
