MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libkernel.la
libkernel_la_SOURCES = \
			Process.h Process.ipp Process.cpp\
			Accelerator.h Accelerator.ipp Accelerator.cpp\
			Descriptor.h Descriptor.ipp\
			Kernel.h Kernel.ipp\
            Mode.h Mode.ipp Mode.cpp\
			Context.h Context.ipp Context.cpp\
            Queue.h Queue.ipp Queue.cpp

export INCLUDES += -I$(abs_srcdir) -I$(abs_builddir)
