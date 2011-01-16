add_subdirectory(allocator)
set(core_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Accelerator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Accelerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Mode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Mode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IOBuffer.cpp
   
    ${CMAKE_CURRENT_SOURCE_DIR}/Context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Context.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Process.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Process.cpp
)
   

# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${core_SRC}
    PARENT_SCOPE)
