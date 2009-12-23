MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = posix 
SUBDIRS = $(OS_DIR) 

noinst_LTLIBRARIES = libmemory.la
libmemory_la_SOURCES = memory.cpp\
			VM.h VM.cpp\
			Map.h Map.cpp\
			PageTable.h PageTable.cpp\
			Manager.h Manager.cpp\
			Region.h Region.cpp\
			Handler.h Handler.cpp\
			BatchManager.h BatchManager.cpp\
			LazyManager.h LazyManager.cpp\
			ProtRegion.h ProtRegion.cpp\
			RollingManager.h RollingManager.cpp\
			RollingRegion.h RollingRegion.cpp
libmemory_la_LIBADD = $(OS_DIR)/libmemory-os.la $(PARAVER_LIB) -lm

BUILT_SOURCES = os

os: $(OS_DIR)
	$(LN_S) -Tf $(srcdir)/$(OS_DIR) os
