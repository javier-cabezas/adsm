MAINTAINERCLEANFILES = Makefile.in

DIST_SUBDIRS = linux
SUBDIRS = linux

noinst_LTLIBRARIES = libparaver.la
libparaver_la_SOURCES = Trace.cpp Record.cpp Element.cpp paraver.cpp
libparaver_la_LIBADD = linux/libparaver-os.la

bin_PROGRAMS = prb2prv
prb2prv_SOURCES = prb2prv.cpp
prb2prv_LDADD = libparaver.la
