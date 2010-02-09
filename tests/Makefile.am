MAINTAINERCLEANFILES=Makefile.in

noinst_PROGRAMS = bandwidth gmacVecAdd\
		gmacTexture gmacConstant\
		gmacMemset gmacMemcpy\
		gmacThreadVecAdd gmacSharedVecAdd\
		gmacStencil gmacThreadStencil\
		gmacMatrixMul\
	 	gmacPingPong gmacCompress gmacCompressSend\
		vmVecAdd vmNull vmStencil vmMatrixMul

bandwidth_SOURCES = bandwidth.cu
bandwidth_LDADD = $(PARAVER_LIB) -L$(CUDA_DIR)/lib$(ARCH) -lcuda -lcudart

gmacVecAdd_SOURCES = gmacVecAdd.cu utils.cpp
gmacVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)


gmacMemset_SOURCES = gmacMemset.cu
gmacMemset_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMemcpy_SOURCES = gmacMemcpy.cu
gmacMemcpy_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacTexture_SOURCES = gmacTexture.cu
gmacTexture_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB) -L$(CUDA_DIR)/lib$(ARCH) -lcudart

gmacConstant_SOURCES = gmacConstant.cu
gmacConstant_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB) -L$(CUDA_DIR)/lib$(ARCH) -lcudart

gmacThreadVecAdd_SOURCES = gmacThreadVecAdd.cu utils.cpp
gmacThreadVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacSharedVecAdd_SOURCES = gmacSharedVecAdd.cu utils.cpp
gmacSharedVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacStencil_DEPENDENCIES = gmacStencilCommon.cu
gmacStencil_SOURCES = gmacStencil.cu utils.cpp
gmacStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacThreadStencil_DEPENDENCIES = gmacStencilCommon.cu
gmacThreadStencil_SOURCES = gmacThreadStencil.cu utils.cpp
gmacThreadStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMatrixMul_DEPENDENCIES = gmacMatrixMulKernel.cu
gmacMatrixMul_SOURCES = gmacMatrixMul.cu utils.cpp
gmacMatrixMul_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacPingPong_SOURCES = gmacPingPong.cu utils.cpp
gmacPingPong_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacCompress_SOURCES = gmacCompress.cu gmacCompressCommon.cu utils.cpp
gmacCompress_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacCompressSend_SOURCES = gmacCompressSend.cu gmacCompressCommon.cu utils.cpp
gmacCompressSend_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmVecAdd_SOURCES = vmVecAdd.cu utils.cpp
vmVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmNull_SOURCES = vmNull.cu utils.cpp
vmNull_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmStencil_DEPENDENCIES = vmStencilCommon.cu
vmStencil_SOURCES = vmStencil.cu utils.cpp
vmStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmMatrixMul_DEPENDENCIES = vmMatrixMulKernel.cu
vmMatrixMul_SOURCES = vmMatrixMul.cu utils.cpp
vmMatrixMul_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

AM_CPPFLAGS = -I$(top_srcdir)/src/gmac -I$(top_builddir)/src/gmac -I$(top_srcdir)/src/

LINK = $(LIBTOOL) --mode=link $(CXX) -static $(CFLAGS) -o $@ 

.cu.o:
	$(NVCC) --keep -v -c $< -o $@ $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

