
include_directories(${KDE4_INCLUDES} ${KDE4_INCLUDE_DIR} ${QT_INCLUDES} )


########### install files ###############

install(FILES  Doxyfile.in DESTINATION )



#original Makefile.am contents follow:

#dist_pkgdata_DATA = Doxyfile.in
#
#doxygen: Doxyfile
#	rm -Rf $(srcdir)/html
#	(cd $(srcdir) ; doxygen $(realpath $(builddir))/Doxyfile)
