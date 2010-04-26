MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libapi.la
libapi_la_SOURCES = opencl.cpp driver.cpp compat.cpp\
	Accelerator.h Accelerator.ipp Accelerator.cpp Error.cpp\
	Kernel.h Kernel.ipp Kernel.cpp    \
	Context.h Context.ipp Context.cpp \
	Program.h Program.ipp Program.cpp
libapi_la_LIBADD = -lOpenCL
