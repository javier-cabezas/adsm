MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libapi.la
libapi_la_SOURCES = cuda.cpp driver.cpp compat.cpp\
	Accelerator.h Accelerator.ipp Accelerator.cpp Error.cpp\
	Context.h Context.cpp\
	Module.h Module.cpp
libapi_la_LIBADD = -lcuda
libapi_la_LDFLAGS = -L$(CUDA_DIR)/lib$(ARCH)
