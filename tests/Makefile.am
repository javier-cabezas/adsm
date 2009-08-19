MAINTAINERCLEANFILES=Makefile.in

noinst_PROGRAMS = bandwidth signal gmacVecAdd\
		gmacTexture gmacConstant\
		gmacThreadVecAdd\
		vmVecAdd

bandwidth_SOURCES = bandwidth.cu
bandwidth_LDADD = $(PARAVER_LIB) -L$(CUDA_DIR)/lib -lcuda -lcudart

signal_SOURCES = signal.cpp
signal_CPPFLAGS = -I$(CUDA_DIR)/include -I$(top_srcdir)/src -I$(top_builddir)/src
signal_LDADD = $(PARAVER_LIB)

gmacVecAdd_SOURCES = gmacVecAdd.cu
gmacVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmVecAdd_SOURCES = vmVecAdd.cu
vmVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)


gmacTexture_SOURCES = gmacTexture.cu
gmacTexture_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB) -lcudart

gmacConstant_SOURCES = gmacConstant.cu
gmacConstant_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB) -lcudart

gmacThreadVecAdd_SOURCES = gmacThreadVecAdd.cu
gmacThreadVecAdd_CPPFLAGS = $(PARAVER_INCLUDE)
gmacThreadVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

AM_CPPFLAGS = -I$(top_srcdir)/src/gmac -I$(top_builddir)/src/gmac -I$(top_srcdir)/src/

LINK = $(LIBTOOL) --mode=link $(CXX) -static $(CFLAGS) -o $@ 

.cu.o:
	$(NVCC) --host-compilation C++ --keep --compiler-bindir=/usr/bin/gcc-4.1 -c $< -o $@ $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

