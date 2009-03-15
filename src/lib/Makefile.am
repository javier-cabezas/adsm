MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = pthread unix
SUBDIRS = unix $(THREAD_DIR)

lib_LTLIBRARIES = libgmac.la
nobase_include_HEADERS = gmac.h gmac/cuda.h gmac/paraver.h
libgmac_la_SOURCES = gmac.cpp cuda.cpp paraver.cpp
libgmac_la_LIBADD = unix/libgmac-io.la $(THREAD_DIR)/libgmac-thread.la ../common/libcommon.la -lcuda -lcudart
libgmac_la_LDFLAGS = -L$(CUDA_DIR)/lib

export INCLUDES += -I$(abs_srcdir) -I$(abs_buildir)
