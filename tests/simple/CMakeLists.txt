# add_subdirectory(cuda)

if(USE_VM)
    add_subdirectory(vm)
else(USE_VM)
    add_subdirectory(gmac)
endif()
