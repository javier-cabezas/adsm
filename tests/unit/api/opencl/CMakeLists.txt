add_subdirectory(hpe)

if(MAKE_LITE)
    add_subdirectory(lite)
endif(MAKE_LITE)

# Export tests one level up
set(unit_SRC ${unit_SRC}
    PARENT_SCOPE)
