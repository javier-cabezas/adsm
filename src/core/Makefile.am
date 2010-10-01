MAINTAINERCLEANFILES = Makefile.in
DIST_SUBDIRS = allocator
SUBDIRS = allocator

noinst_LTLIBRARIES = libcore.la
libcore_la_SOURCES = \
			Process.h Process.ipp Process.cpp\
			Accelerator.h Accelerator.ipp Accelerator.cpp\
			Descriptor.h \
			Kernel.h Kernel.cpp\
            IOBuffer.h \
            Mode.h Mode.ipp Mode.cpp\
			Context.h Context.cpp\
            Queue.h Queue.cpp

libcore_la_LIBADD = allocator/libcore-allocator.la

export INCLUDES += -I$(abs_srcdir) -I$(abs_builddir)
