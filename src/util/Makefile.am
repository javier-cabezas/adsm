MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libutil.la
libutil_la_SOURCES = Lock.h Lock.ipp Lock.cpp\
					 Semaphore.h Semaphore.ipp Semaphore.cpp\
                     Parameter.h Parameter.ipp Parameter.cpp Parameter.def

export INCLUDES += -I$(abs_srcdir) -I$(abs_builddir)
