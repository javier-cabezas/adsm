MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = posix 
SUBDIRS = $(OS_DIR) 

noinst_LTLIBRARIES = libmemory.la
libmemory_la_SOURCES = memory.cpp\
			MemManager.h MemManager.cpp\
			MemRegion.h MemRegion.cpp\
			MemHandler.h MemHandler.cpp\
			BatchManager.h BatchManager.cpp\
			LazyManager.h LazyManager.cpp\
			ProtRegion.h\
			RollingManager.h RollingManager.cpp\
			RollingRegion.h RollingRegion.cpp
libmemory_la_LIBADD = $(OS_DIR)/libmemory-os.la $(PARAVER_LIB)

BUILT_SOURCES = os

os: $(OS_DIR)
	$(LN_S) -Tf $(srcdir)/$(OS_DIR) os
