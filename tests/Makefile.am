MAINTAINERCLEANFILES=Makefile.in

noinst_PROGRAMS = bandwidth threadBandwidth\
				  hybridComputation gmacHybridComputation gmacVecAdd\
		gmacTexture gmacConstant\
		gmacMemset gmacMemcpy\
		gmacThreadVecAdd gmacSharedVecAdd\
		gmacStencil gmacThreadStencil\
		gmacMatrixMul\
	 	gmacPingPong gmacCompress gmacCompressSend\
		gmacMigration\
		vmVecAdd vmNull vmStencil vmMatrixMul

noinst_SCRIPTS = gmacTests.py

EXTRA_DIST = gmacTests.py gmacLaunch.sh\
    gmacStencilCommon.cu vmStencilCommon.cu\
    gmacMatrixMulKernel.cu vmMatrixMulKernel.cu

GMAC_TESTSOURCES=debug.h utils.cpp utils.ipp utils.h

bandwidth_SOURCES = bandwidth.cu $(GMAC_TESTSOURCES)
bandwidth_LDADD = $(PARAVER_LIB) -L$(CUDA_DIR)/lib -L$(CUDA_DIR)/lib$(ARCH) -lcuda -lcudart

threadBandwidth_SOURCES = threadBandwidth.cu barrier.c barrier.h $(GMAC_TESTSOURCES)
threadBandwidth_LDADD = $(PARAVER_LIB) -L$(CUDA_DIR)/lib -L$(CUDA_DIR)/lib$(ARCH) -lcuda -lcudart

hybridComputation_SOURCES = hybridComputation.cu $(GMAC_TESTSOURCES)
hybridComputation_LDADD = $(PARAVER_LIB) -L$(CUDA_DIR)/lib -L$(CUDA_DIR)/lib$(ARCH) -lcuda -lcudart

gmacVecAdd_SOURCES = gmacVecAdd.cu $(GMAC_TESTSOURCES)
gmacVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMemset_SOURCES = gmacMemset.cu $(GMAC_TESTSOURCES)
gmacMemset_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMemcpy_SOURCES = gmacMemcpy.cu $(GMAC_TESTSOURCES)
gmacMemcpy_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacTexture_SOURCES = gmacTexture.cu $(GMAC_TESTSOURCES)
gmacTexture_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacConstant_SOURCES = gmacConstant.cu $(GMAC_TESTSOURCES)
gmacConstant_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacThreadVecAdd_SOURCES = gmacThreadVecAdd.cu $(GMAC_TESTSOURCES)
gmacThreadVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacSharedVecAdd_SOURCES = gmacSharedVecAdd.cu $(GMAC_TESTSOURCES)
gmacSharedVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacStencil_DEPENDENCIES = gmacStencilCommon.cu
gmacStencil_SOURCES = gmacStencil.cu barrier.c barrier.h $(GMAC_TESTSOURCES)
gmacStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacThreadStencil_DEPENDENCIES = gmacStencilCommon.cu
gmacThreadStencil_SOURCES = gmacThreadStencil.cu barrier.c barrier.h $(GMAC_TESTSOURCES)
gmacThreadStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMatrixMul_DEPENDENCIES = gmacMatrixMulKernel.cu
gmacMatrixMul_SOURCES = gmacMatrixMul.cu $(GMAC_TESTSOURCES)
gmacMatrixMul_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacPingPong_SOURCES = gmacPingPong.cu $(GMAC_TESTSOURCES)
gmacPingPong_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacCompress_SOURCES = gmacCompress.h gmacCompress.cu gmacCompressCommon.cu $(GMAC_TESTSOURCES)
gmacCompress_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacCompressSend_SOURCES = gmacCompress.h gmacCompressSend.cu gmacCompressCommon.cu barrier.c barrier.h $(GMAC_TESTSOURCES)
gmacCompressSend_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacMigration_SOURCES = gmacMigration.cu $(GMAC_TESTSOURCES)
gmacMigration_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

gmacHybridComputation_SOURCES = gmacHybridComputation.cu $(GMAC_TESTSOURCES)
gmacHybridComputation_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)
gmacHybridComputation_LDFLAGS = -fopenmp

vmVecAdd_SOURCES = vmVecAdd.cu $(GMAC_TESTSOURCES)
vmVecAdd_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmNull_SOURCES = vmNull.cu $(GMAC_TESTSOURCES)
vmNull_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmStencil_DEPENDENCIES = vmStencilCommon.cu
vmStencil_SOURCES = vmStencil.cu barrier.c barrier.h $(GMAC_TESTSOURCES)
vmStencil_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

vmMatrixMul_DEPENDENCIES = vmMatrixMulKernel.cu
vmMatrixMul_SOURCES = vmMatrixMul.cu $(GMAC_TESTSOURCES)
vmMatrixMul_LDADD = ../src/gmac/libgmac.la $(PARAVER_LIB)

AM_CPPFLAGS = -I$(top_srcdir)/src/gmac -I$(top_builddir)/src/gmac -I$(top_srcdir)/src/

LINK = $(LIBTOOL) --mode=link $(CXX) -static $(CFLAGS) -o $@ 

gmacHybridComputation.o: gmacHybridComputation.cu
	$(NVCC) -c $< -o $@ $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CUDA_CFLAGS) -Xcompiler -fopenmp

.cu.o:
	$(NVCC) -c $< -o $@ $($(basename $@)_CPPFLAGS) $($(basename $@)_CUDAFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CUDA_CFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

