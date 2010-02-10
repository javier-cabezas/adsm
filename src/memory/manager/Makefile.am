MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libmemory-manager.la
libmemory_manager_la_SOURCES = BatchManager.h BatchManager.ipp BatchManager.cpp\
                               LazyManager.h  LazyManager.ipp  LazyManager.cpp\
							   RollingRegion.h RollingRegion.ipp RollingRegion.cpp\
							   RollingManager.h RollingManager.ipp RollingManager.cpp

export INCLUDES += -I$(abs_srcdir)/.. -I$(abs_builddir)/..
