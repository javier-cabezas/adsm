MAINTAINERCLEANFILES = Makefile.in

SUBDIRS = $(ACC_DIR) $(PARAVER_DIR) memory gmac
DIST_SUBDIRS = config memory paraver gmac

BUILT_SOURCES = os acc

export INCLUDES := -I$(CUDA_DIR)/include -I$(abs_srcdir) -I$(abs_builddir)

os: $(OS_DIR)
	$(LN_S) -Tf $(srcdir)/$(OS_DIR) os

acc: $(ACC_DIR)
	$(LN_S) -Tf $(srcdir)/$(ACC_DIR) acc
