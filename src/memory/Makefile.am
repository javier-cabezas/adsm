MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = protocol allocator posix 
SUBDIRS = protocol allocator $(OS_DIR) 

noinst_LTLIBRARIES = libmemory.la
libmemory_la_SOURCES = memory.cpp\
            Bitmap.h Bitmap.ipp Bitmap.cpp\
            Block.h Block.ipp\
            Object.h Object.ipp Object.cpp ObjectSet.h\
            StateObject.h StateObject.ipp\
            SharedObject.h SharedObject.ipp DistributedObject.h\
            ReplicatedObject.h ReplicatedObject.ipp CentralizedObject.h \
			Map.h Map.cpp\
            Protocol.h Protocol.cpp\
			Manager.h Manager.cpp Manager.ipp\
			Handler.h Handler.cpp\
            Allocator.h

libmemory_la_LIBADD = protocol/libmemory-protocol.la allocator/libmemory-allocator.la $(OS_DIR)/libmemory-os.la $(PARAVER_LIB) -lm

BUILT_SOURCES = os

os: $(OS_DIR)
	$(LN_S) $(LN_FLAGS) $(srcdir)/$(OS_DIR) os
