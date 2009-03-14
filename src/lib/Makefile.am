MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = pthread
SUBDIRS = $(THREAD_DIR)

lib_LTLIBRARIES = libgmac.la
nobase_include_HEADERS = gmac.h gmac/cuda.h
libgmac_la_SOURCES = gmac.cpp cuda.cpp paraver.cpp
libgmac_la_LIBADD = $(THREAD_DIR)/libgmac-thread.la ../common/libcommon.la -lcuda -lcudart
libgmac_la_LDFLAGS = -L$(CUDA_DIR)/lib

export INCLUDES += -I$(abs_srcdir) -I$(abs_buildir)
