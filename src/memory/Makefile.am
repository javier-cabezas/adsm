MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = posix 
SUBDIRS = $(OS_DIR) 

noinst_LTLIBRARIES = libmemory.la
libmemory_la_SOURCES = \
			MemManager.h MemManager.cpp\
			MemRegion.h\
			BatchManager.h BatchManager.cpp\
			LazyManager.h LazyManager.cpp\
			MemHandler.h MemHandler.cpp\
			ProtRegion.h ProtRegion.cpp\
			CacheManager.h CacheManager.cpp\
			CacheRegion.h CacheRegion.cpp
libmemory_la_LIBADD = $(OS_DIR)/libmemory-os.la $(PARAVER_LIB)

BUILT_SOURCES = os

export INCLUDES += -I$(abs_srcdir) -I$(abs_builddir)

os: $(OS_DIR)
	$(LN_S) -Tf $(srcdir)/$(OS_DIR) os
