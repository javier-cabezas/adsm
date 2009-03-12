MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = unix
SUBDIRS = unix

noinst_LTLIBRARIES = libcommon.la
libcommon_la_SOURCES = config.cpp\
			MemManager.h MemManager.cpp\
			MemRegion.h MemRegion.cpp\
			BatchManager.h BatchManager.cpp\
			LazyManager.h LazyManager.cpp\
			CacheManager.h CacheManager.cpp\
			StatManager.h StatManager.cpp
libcommon_la_LIBADD = unix/libcommon.la $(PARAVER_LIB)
