add_subdirectory(${OS_DIR})
if(USE_TRACE_PARAVER)
    add_subdirectory(paraver)
endif(USE_TRACE_PARAVER)

set(trace_SRC
    Tracer.h Tracer-impl.h Tracer.cpp
    Console.h Console.cpp
    Paraver.h Paraver.cpp)

add_gmac_sources(gmac ${trace_SRC})
