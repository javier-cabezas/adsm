MAINTAINERCLEANFILES = Makefile.in

DIST_SUBDIRS = posix
SUBDIRS = $(OS_DIR)

noinst_LTLIBRARIES = libutil.la
libutil_la_SOURCES = Lock.h Semaphore.h Private.h\
                     Parameter.h Parameter.ipp Parameter.cpp Parameter.def\
                     ReusableObject.h

libutil_la_LIBADD = $(OS_DIR)/libutil-os.la

BUILT_SOURCES = os

os: $(OS_DIR)
	$(LN_S) $(LN_FLAGS) $(srcdir)/$(OS_DIR) os
