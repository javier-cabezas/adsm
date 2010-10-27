# Files required to test gmac::memory::Memory
set(memory_SRC
    ${CMAKE_SOURCE_DIR}/src/memory/${OS_DIR}/FileMap.h
    ${CMAKE_SOURCE_DIR}/src/memory/${OS_DIR}/FileMap.cpp
    ${CMAKE_SOURCE_DIR}/src/memory/Memory.h
    ${CMAKE_SOURCE_DIR}/src/memory/${OS_DIR}/Memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Memory.cpp)

# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${memory_SRC}
    PARENT_SCOPE)
