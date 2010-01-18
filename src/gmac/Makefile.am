MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = stdc posix pthreads
SUBDIRS = stdc $(OS_DIR) $(THREAD_DIR)

lib_LTLIBRARIES = libgmac.la
nobase_include_HEADERS = gmac.h gmac/paraver.h gmac/vm.h
libgmac_la_SOURCES = gmac/paraver.h gmac.h gmac.cpp paraver.cpp
libgmac_la_LIBADD = stdc/libgmac-stdc.la $(OS_DIR)/libgmac-os.la $(THREAD_DIR)/libgmac-thread.la\
		../kernel/libkernel.la\
		../memory/libmemory.la\
		../config/libconfig.la\
		../util/libutil.la\
		../api/$(API_DIR)/libapi.la\
		$(PARAVER_LIB)

libgmac_la_LDFLAGS = -L$(CUDA_DIR)/lib64 -ldl

BUILT_SOURCES = os

export INCLUDES += -I$(abs_srcdir) -I$(abs_builddir) -I$(abs_srcdir)/../memory -I$(abs_builddir)/../memory

os: $(OS_DIR)
	$(LN_S) -sTf $(srcdir)/$(OS_DIR) os
