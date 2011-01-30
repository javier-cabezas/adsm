set(paraver_SRC
    Lock.h ${OS_DIR}/Lock.h ${OS_DIR}/Lock-impl.h
    Element.h Element-impl.h Element.cpp
    Names.h Names.cpp
    Record.h Record.cpp
    Trace.h Trace.cpp
)

add_gmac_sources(gmac ${paraver_SRC})
    

#set(prb2prv_SRC ${CMAKE_CURRENT_SOURCE_DIR}/prb2prv.cpp)
#add_executable(prb2prv ${prb2prv_SRC} ${paraven_SRC})
#set_target_properties(prb2prv PROPERTIES COMPILE_DEFINITIONS STATIC)

#set(makepcf_SRC ${CMAKE_CURRENT_SOURCE_DIR}/makepcf.cpp)
#add_executable(makepcf ${makepcf_SRC} ${paraver_SRC})
#target_link_libraries(makepcf ${lib_LIBS})
#set_target_properties(makepcf PROPERTIES COMPILE_DEFINITIONS STATIC)

#install(TARGETS prb2prv makepcf
#    RUNTIME DESTINATION bin)
