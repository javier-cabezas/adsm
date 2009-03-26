MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = stdc posix pthread 
SUBDIRS = stdc $(OS_DIR) $(THREAD_DIR)

lib_LTLIBRARIES = libgmac.la
nobase_include_HEADERS = gmac.h gmac/cuda.h gmac/paraver.h
libgmac_la_SOURCES = gmac.cpp paraver.cpp
libgmac_la_LIBADD = stdc/libgmac-stdc.la $(OS_DIR)/libgmac-os.la $(THREAD_DIR)/libgmac-thread.la\
		../memory/libmemory.la\
		../$(ACC_DIR)/libacc.la
libgmac_la_LDFLAGS = -L$(CUDA_DIR)/lib

BUILT_SOURCES = os

export INCLUDES += -I$(abs_srcdir) -I$(abs_builddir) -I$(abs_srcdir)/../memory -I$(abs_builddir)/../memory

os: $(OS_DIR)
	$(LN_S) -Tf $(srcdir)/$(OS_DIR) os
