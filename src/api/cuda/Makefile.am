MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libapi.la
libapi_la_SOURCES = cuda.cpp driver.cpp compat.cpp\
	Accelerator.h Accelerator.ipp Accelerator.cpp Error.cpp\
	Kernel.h Kernel.ipp Kernel.cpp    \
    IOBuffer.h IOBuffer.cpp\
    Mode.h Mode.ipp Mode.cpp\
	Context.h Context.ipp Context.cpp \
	Module.h Module.ipp Module.cpp\
    CentralizedObject.cpp Bitmap.cpp
libapi_la_LIBADD = -lcuda
libapi_la_LDFLAGS = -L$(CUDA_DIR)/lib$(ARCH)
