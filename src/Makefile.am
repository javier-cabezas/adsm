MAINTAINERCLEANFILES = Makefile.in

SUBDIRS = util trace memory core os api config gmac
DIST_SUBDIRS = util trace config api os memory core gmac

BUILT_SOURCES = config/os config/api

export INCLUDES += -I$(CUDA_DIR)/include\
	-I$(abs_srcdir) -I$(abs_builddir)\
	-I$(abs_srcdir)/config -I$(abs_builddir)/config\
	-I$(abs_srcdir)/util   -I$(abs_builddir)/util\
	$(PARAVER_INCLUDE)

config/os: os/$(OS_DIR)
	$(LN_S) $(LN_FLAGS) $(abs_srcdir)/os/$(OS_DIR) config/os

config/api: api/$(API_DIR)
	$(LN_S) $(LN_FLAGS) $(abs_srcdir)/api/$(API_DIR) config/api
