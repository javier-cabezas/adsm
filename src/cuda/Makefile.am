MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libacc.la
libacc_la_SOURCES = cuda.cpp
libacc_la_LIBADD = -lcudart -lcuda
libacc_la_LDFLAGS = -L$(CUDA_DIR)/lib
