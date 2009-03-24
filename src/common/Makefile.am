MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = unix
SUBDIRS = $(OS_DIR)

noinst_LTLIBRARIES = libcommon.la
libcommon_la_DEPENDENCIES = os
libcommon_la_SOURCES = config.cpp\
			MemManager.h MemManager.cpp\
			MemRegion.h MemRegion.cpp\
			BatchManager.h BatchManager.cpp\
			LazyManager.h LazyManager.cpp\
			CacheManager.h CacheManager.cpp
libcommon_la_LIBADD = $(OS_DIR)/libcommon.la $(PARAVER_LIB)

os: $(OS_DIR)
	$(LN_S) $(srcdir)/$(OS_DIR) os
