dist_pkgdata_DATA = Doxyfile.in

doxygen: Doxyfile
	rm -Rf $(srcdir)/html
	(cd $(srcdir) ; doxygen $(realpath $(builddir))/Doxyfile)
