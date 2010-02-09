MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libapi.la
libapi_la_SOURCES = cuda.cpp \
	GPU.h GPU.cpp \
	Context.h Context.cpp\
	Error.cpp

libapi_la_LIBADD = -lcudart -lcuda
libapi_la_LDFLAGS = -L$(CUDA_DIR)/lib$(ARCH)
