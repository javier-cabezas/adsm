########### install files ###############

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND doxygen Doxyfile
    RESULT_VARIABLE ret_var
    OUTPUT_VARIABLE output_var)

#install(FILES  Doxyfile.in DESTINATION )

#original Makefile.am contents follow:

#dist_pkgdata_DATA = Doxyfile.in
#
#doxygen: Doxyfile
#	rm -Rf $(srcdir)/html
#	(cd $(srcdir) ; doxygen $(realpath $(builddir))/Doxyfile)
