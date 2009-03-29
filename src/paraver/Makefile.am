MAINTAINERCLEANFILES = Makefile.in

DIST_SUBDIRS = posix
SUBDIRS = $(OS_DIR)

noinst_LTLIBRARIES = libparaver.la
libparaver_la_includedir = $(includedir)/paraver
libparaver_la_include_HEADERS = Types.h Names.h
libparaver_la_SOURCES = Trace.cpp Record.cpp Element.cpp Types.cpp Pcf.cpp paraver.cpp
libparaver_la_LIBADD = $(OS_DIR)/libparaver-os.la

bin_PROGRAMS = prb2prv makepcf
prb2prv_SOURCES = prb2prv.cpp
prb2prv_LDADD = libparaver.la

makepcf_SOURCES = makepcf.cpp
makepcf_LDADD = libparaver.la
