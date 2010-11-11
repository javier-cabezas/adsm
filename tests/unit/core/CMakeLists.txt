add_subdirectory(allocator)
set(core_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/IOBuffer.cpp)

# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${core_SRC}
    PARENT_SCOPE)
