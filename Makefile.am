MAINTAINERCLEANFILES = aclocal.m4 config.guess config.sub configure COPYING\
depcomp INSTALL install-sh ltmain.sh Makefile.in missing

if BUILD_TESTS
TESTS_DIR = tests
endif

SUBDIRS = lib src $(TESTS_DIR)
DIST_SUBDIRS = lib src doc tests

doxygen:
	make -C doc doxygen
