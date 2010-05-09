MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = manager posix 
SUBDIRS = manager $(OS_DIR) 

noinst_LTLIBRARIES = libmemory.la
libmemory_la_SOURCES = memory.cpp\
			VM.h VM.ipp\
			Map.h Map.ipp Map.cpp\
			PageTable.h PageTable.ipp PageTable.cpp\
			Manager.h Manager.ipp Manager.cpp\
			Region.h Region.ipp Region.cpp\
			Handler.h Handler.cpp\
			ProtRegion.h ProtRegion.ipp ProtRegion.cpp

libmemory_la_LIBADD = manager/libmemory-manager.la $(OS_DIR)/libmemory-os.la $(PARAVER_LIB) -lm

BUILT_SOURCES = os

os: $(OS_DIR)
	$(LN_S) $(LN_FLAGS) $(srcdir)/$(OS_DIR) os
