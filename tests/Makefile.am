MAINTAINERCLEANFILES=Makefile.in

noinst_PROGRAMS = bandwidth gmacVecAdd\
		gmacTexture gmacConstant\
		gmacMemset gmacMemcpy\
		gmacThreadVecAdd\
		gmacStencil\
		gmacThreadStencil\
		vmVecAdd vmNull

bandwidth_SOURCES = bandwidth.cu
bandwidth_LDADD = $(PARAVER_LIB) -L$(CUDA_DIR)/lib -lcuda -lcudart

gmacVecAdd_SOURCES = gmacVecAdd.cu utils.c
gmacVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmVecAdd_SOURCES = vmVecAdd.cu utils.c
vmVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmNull_SOURCES = vmNull.cu
vmNull_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMemset_SOURCES = gmacMemset.cu
gmacMemset_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMemcpy_SOURCES = gmacMemcpy.cu
gmacMemcpy_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacTexture_SOURCES = gmacTexture.cu
gmacTexture_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB) -lcudart

gmacConstant_SOURCES = gmacConstant.cu
gmacConstant_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB) -lcudart

gmacThreadVecAdd_SOURCES = gmacThreadVecAdd.cu utils.c
gmacThreadVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacStencil_DEPENDENCIES = gmacStencilCommon.cu
gmacStencil_SOURCES = gmacStencil.cu utils.c
gmacStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacThreadStencil_DEPENDENCIES = gmacStencilCommon.cu
gmacThreadStencil_SOURCES = gmacThreadStencil.cu utils.c
gmacThreadStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

AM_CPPFLAGS = -I$(top_srcdir)/src/gmac -I$(top_builddir)/src/gmac -I$(top_srcdir)/src/

LINK = $(LIBTOOL) --mode=link $(CXX) -static $(CFLAGS) -o $@ 

.cu.o:
	$(NVCC) --host-compilation C++ --keep --compiler-bindir=/usr/bin/gcc-4.1 -c $< -o $@ $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

