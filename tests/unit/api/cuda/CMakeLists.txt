set(accelerator_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/Accelerator.cpp)

# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${accelerator_SRC}
    PARENT_SCOPE)
