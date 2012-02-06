set(dsm_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/manager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mapping.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mapping.cpp
)  


# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${dsm_SRC}
    PARENT_SCOPE)
