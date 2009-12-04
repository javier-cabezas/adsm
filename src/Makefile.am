MAINTAINERCLEANFILES = Makefile.in

SUBDIRS = util memory kernel os api gmac
DIST_SUBDIRS = config util api os memory kernel gmac

BUILT_SOURCES = config/os config/api

export INCLUDES += -I$(CUDA_DIR)/include\
	-I$(abs_srcdir) -I$(abs_builddir)\
	-I$(abs_srcdir)/config -I$(abs_builddir)/config\
	$(PARAVER_INCLUDE)

config/os: os/$(OS_DIR)
	$(LN_S) -Tf $(abs_srcdir)/os/$(OS_DIR) config/os

config/api: api/$(API_DIR)
	$(LN_S) -Tf $(abs_srcdir)/api/$(API_DIR) config/api
