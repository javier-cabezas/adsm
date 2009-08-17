MAINTAINERCLEANFILES = Makefile.in

SUBDIRS = $(PARAVER_DIR) api os memory gmac
DIST_SUBDIRS = config paraver api os memory gmac

BUILT_SOURCES = config/os config/api

export INCLUDES := -I$(CUDA_DIR)/include -I$(abs_srcdir)/config -I$(abs_builddir)/config\
	-I$(abs_srcdir) -I$(abs_builddir)

config/os: os/$(OS_DIR)
	$(LN_S) -Tf ../$(srcdir)/os/$(OS_DIR) config/os

config/api: api/$(API_DIR)
	$(LN_S) -Tf ../$(srcdir)/api/$(API_DIR) config/api
