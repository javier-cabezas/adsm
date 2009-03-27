MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libacc.la
libacc_la_SOURCES = cuda.cpp driver.cpp
libacc_la_LIBADD = -lcuda
libacc_la_LDFLAGS = -L$(CUDA_DIR)/lib
