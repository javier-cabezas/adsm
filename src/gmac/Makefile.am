MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = stdc posix pthreads
SUBDIRS = stdc $(OS_DIR) $(THREAD_DIR)

lib_LTLIBRARIES = libgmac.la
nobase_include_HEADERS = gmac.h gmac/vm.h
libgmac_la_SOURCES = gmac.h gmac.cpp paraver.cpp init.h
libgmac_la_LIBADD = stdc/libgmac-stdc.la $(OS_DIR)/libgmac-os.la $(THREAD_DIR)/libgmac-thread.la\
		../kernel/libkernel.la\
		../memory/libmemory.la\
		../util/libutil.la\
		../api/$(API_DIR)/libapi.la\
		$(PARAVER_LIB)

libgmac_la_LDFLAGS = -L$(CUDA_DIR)/lib -L$(CUDA_DIR)/lib$(ARCH) -ldl

BUILT_SOURCES = os

export INCLUDES += -I$(abs_srcdir) -I$(abs_builddir) -I$(abs_srcdir)/../memory -I$(abs_builddir)/../memory

os: $(OS_DIR)
	$(LN_S) $(LN_FLAGS) $(srcdir)/$(OS_DIR) os
