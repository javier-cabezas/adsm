MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = allocator
SUBDIRS = allocator

noinst_LTLIBRARIES = libkernel.la
libkernel_la_SOURCES = \
			Process.h Process.cpp\
			Accelerator.h Accelerator.cpp\
			Descriptor.h \
			Kernel.h Kernel.cpp\
            IOBuffer.h \
            Mode.h Mode.cpp\
			Context.h Context.cpp\
            Queue.h Queue.cpp

libkernel_la_LIBADD = allocator/libkernel-allocator.la

export INCLUDES += -I$(abs_srcdir) -I$(abs_builddir)
