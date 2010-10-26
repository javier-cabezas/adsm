function(add_gmac_sources)

    # Add the VC group to the list
    get_property(gmac_GROUPS GLOBAL PROPERTY gmac_GROUPS)
    string(REPLACE ${CMAKE_SOURCE_DIR} "" __group "${CMAKE_CURRENT_SOURCE_DIR}")
    string(REGEX REPLACE "^/" "" __group "${__group}")
    string(REPLACE "/" "\\\\" __group "${__group}")
    set(gmac_GROUPS ${gmac_GROUPS} ${__group})
    set_property(GLOBAL PROPERTY gmac_GROUPS ${gmac_GROUPS})

    foreach(__file ${ARGV})
        set(__source __source-NOTFOUND)
        find_file(__source ${__file} ${CMAKE_CURRENT_SOURCE_DIR} NO_DEFAULT_PATH)
        if(EXISTS ${__source})
            set(__sources ${__sources} ${__source})
        else()
            set(__binary __binary-NOTFOUND)
            find_file(__binary ${__file} ${CMAKE_CURRENT_BINARY_DIR} NO_DEFAULT_PATH)
            if(EXISTS ${__binary})
                set(__sources ${__sources} ${__binary})
            endif()
        endif()
    endforeach()

    # Add the source files to the global list
    get_property(gmac_SRC GLOBAL PROPERTY gmac_SRC)
    set(gmac_SRC ${gmac_SRC} ${__sources})
    set_property(GLOBAL PROPERTY gmac_SRC ${gmac_SRC})

    # Bind source files to the group
    string(REPLACE "/" "_" __group_name "${__group}")
    set_property(GLOBAL PROPERTY gmac_GROUP${__group_name} ${__sources})
endfunction(add_gmac_sources)

function(configure_gmac_groups)
    get_property(gmac_GROUPS GLOBAL PROPERTY gmac_GROUPS)

    foreach(__group ${gmac_GROUPS})
        string(REPLACE "/" "_" __group_name "${__group}")
        get_property(__files GLOBAL PROPERTY gmac_GROUP${__group_name})
        source_group(${__group} FILES ${__files})
    endforeach()
endfunction(configure_gmac_groups)


function(add_gmac_test_include)
    get_property(gmac_test_INCLUDE GLOBAL PROPERTY gmac_test_INCLUDE)
    set(gmac_test_INCLUDE ${gmac_test_INCLUDE} ${ARGV})
    set_property(GLOBAL PROPERTY gmac_test_INCLUDE ${gmac_test_INCLUDE})
endfunction(add_gmac_test_include)


function(add_gmac_test_library)
    get_property(gmac_test_LIB GLOBAL PROPERTY gmac_test_LIB)
    set(gmac_test_LIB ${gmac_test_LIB} ${ARGV})
    set_property(GLOBAL PROPERTY gmac_test_LIB ${gmac_test_LIB})
endfunction(add_gmac_test_library)

macro(import_gmac_libraries)
    get_property(gmac_test_INCLUDE GLOBAL PROPERTY gmac_test_INCLUDE)
    get_property(gmac_test_LIB GLOBAL PROPERTY gmac_test_LIB)
endmacro(import_gmac_libraries)

