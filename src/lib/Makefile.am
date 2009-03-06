MAINTAINERCLEANFILES = Makefile.in

lib_LTLIBRARIES = libgmac.la
nobase_include_HEADERS = gmac.h gmac/cuda.h
libgmac_la_SOURCES = gmac.cpp cuda.cpp
libgmac_la_LIBADD = ../common/libcommon.la -lcuda -lcudart
libgmac_la_LDFLAGS = -L$(CUDA_DIR)/lib
