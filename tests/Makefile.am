MAINTAINERCLEANFILES=Makefile.in

noinst_PROGRAMS = bandwidth gmacVecAdd\
		gmacTexture gmacConstant\
		gmacMemset gmacMemcpy\
		gmacThreadVecAdd gmacSharedVecAdd\
		gmacStencil gmacThreadStencil\
		gmacMatrixMul\
	 	gmacPingPong gmacCompress gmacCompressSend\
		vmVecAdd vmNull vmStencil vmMatrixMul

GMAC_TESTSOURCES=test.h test.c

bandwidth_SOURCES = bandwidth.cu $(GMAC_TESTSOURCES)
bandwidth_LDADD = $(PARAVER_LIB) -L$(CUDA_DIR)/lib -L$(CUDA_DIR)/lib$(ARCH) -lcuda -lcudart

gmacVecAdd_SOURCES = gmacVecAdd.cu utils.cpp $(GMAC_TESTSOURCES)
gmacVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMemset_SOURCES = gmacMemset.cu $(GMAC_TESTSOURCES)
gmacMemset_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMemcpy_SOURCES = gmacMemcpy.cu $(GMAC_TESTSOURCES)
gmacMemcpy_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacTexture_SOURCES = gmacTexture.cu $(GMAC_TESTSOURCES)
gmacTexture_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB) -L$(CUDA_DIR)/lib$(ARCH) -lcudart

gmacConstant_SOURCES = gmacConstant.cu $(GMAC_TESTSOURCES)
gmacConstant_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB) -L$(CUDA_DIR)/lib$(ARCH) -lcudart

gmacThreadVecAdd_SOURCES = gmacThreadVecAdd.cu utils.cpp $(GMAC_TESTSOURCES)
gmacThreadVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacSharedVecAdd_SOURCES = gmacSharedVecAdd.cu utils.cpp $(GMAC_TESTSOURCES)
gmacSharedVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacStencil_DEPENDENCIES = gmacStencilCommon.cu
gmacStencil_SOURCES = gmacStencil.cu utils.cpp $(GMAC_TESTSOURCES)
gmacStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacThreadStencil_DEPENDENCIES = gmacStencilCommon.cu
gmacThreadStencil_SOURCES = gmacThreadStencil.cu utils.cpp $(GMAC_TESTSOURCES)
gmacThreadStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMatrixMul_DEPENDENCIES = gmacMatrixMulKernel.cu
gmacMatrixMul_SOURCES = gmacMatrixMul.cu utils.cpp $(GMAC_TESTSOURCES)
gmacMatrixMul_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacPingPong_SOURCES = gmacPingPong.cu utils.cpp $(GMAC_TESTSOURCES)
gmacPingPong_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacCompress_SOURCES = gmacCompress.cu gmacCompressCommon.cu utils.cpp $(GMAC_TESTSOURCES)
gmacCompress_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacCompressSend_SOURCES = gmacCompressSend.cu gmacCompressCommon.cu utils.cpp $(GMAC_TESTSOURCES)
gmacCompressSend_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmVecAdd_SOURCES = vmVecAdd.cu utils.cpp $(GMAC_TESTSOURCES)
vmVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmNull_SOURCES = vmNull.cu utils.cpp $(GMAC_TESTSOURCES)
vmNull_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmStencil_DEPENDENCIES = vmStencilCommon.cu
vmStencil_SOURCES = vmStencil.cu utils.cpp $(GMAC_TESTSOURCES)
vmStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmMatrixMul_DEPENDENCIES = vmMatrixMulKernel.cu
vmMatrixMul_SOURCES = vmMatrixMul.cu utils.cpp $(GMAC_TESTSOURCES)
vmMatrixMul_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

AM_CPPFLAGS = -I$(top_srcdir)/src/gmac -I$(top_builddir)/src/gmac -I$(top_srcdir)/src/

LINK = $(LIBTOOL) --mode=link $(CXX) -static $(CFLAGS) -o $@ 

.cu.o:
	$(NVCC) -c $< -o $@ $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

