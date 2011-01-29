configure_file(gmac_vec_add.in gmac_vec_add)
configure_file(gmac_thread_vec_add.in gmac_thread_vec_add)

install(FILES Atffile DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac/atf)
install(PROGRAMS
        ${CMAKE_CURRENT_BINARY_DIR}/gmac_vec_add
        ${CMAKE_CURRENT_BINARY_DIR}/gmac_thread_vec_add
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac/atf)
