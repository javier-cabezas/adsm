set(cuda_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Accelerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Mode.cpp)

# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${cuda_SRC}
    PARENT_SCOPE)
