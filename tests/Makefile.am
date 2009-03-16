MAINTAINERCLEANFILES=Makefile.in

noinst_PROGRAMS = bandwidth vecAdd gmacVecAdd gmacThreadVecAdd

bandwidth_SOURCES = bandwidth.cu
bandwidth_LDADD = -L$(CUDA_DIR)/lib -lcuda -lcudart

vecAdd_SOURCES = vecAdd.cu
vecAdd_LDADD = -L$(CUDA_DIR)/lib -lcuda -lcudart

gmacVecAdd_SOURCES = gmacVecAdd.cu
gmacVecAdd_LDADD = ../src/lib/libgmac.la $(PARAVER_LIB)

gmacThreadVecAdd_SOURCES = gmacThreadVecAdd.cu
gmacThreadVecAdd_LDADD = ../src/lib/libgmac.la $(PARAVER_LIB)

AM_CPPFLAGS = -I$(top_srcdir)/src/lib -I$(top_srcdir)/src -I$(top_builddir)/src

LINK = $(LIBTOOL) --mode=link $(CC) $(CFLAGS) -o $@ 

.cu.o:
	$(NVCC) --host-compilation C++ --compiler-bindir=/usr/bin/gcc-4.1 -c $< -o $@ $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

