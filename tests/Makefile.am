MAINTAINERCLEANFILES=Makefile.in

bin_PROGRAMS = vecAdd gmacVecAdd gmacThreadVecAdd

vecAdd_SOURCES = vecAdd.cu
vecAdd_LDADD = -L$(CUDA_DIR)/lib -lcuda -lcudart

gmacVecAdd_SOURCES = gmacVecAdd.cu
gmacVecAdd_CPPFLAGS = -I$(top_srcdir)/src/lib
gmacVecAdd_LDADD = ../src/lib/libgmac.la ../src/paraver/libparaver.la

gmacThreadVecAdd_SOURCES = gmacThreadVecAdd.cu
gmacThreadVecAdd_CPPFLAGS = -I$(top_srcdir)/src/lib
gmacThreadVecAdd_LDADD = ../src/lib/libgmac.la ../src/paraver/libparaver.la


LINK = $(LIBTOOL) --mode=link $(CC) -static $(CFLAGS) -o $@ 

.cu.o:
	$(NVCC) --host-compilation C -c $< -o $@ $($(basename $@)_CPPFLAGS) $(CFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS) $(CFLAGS)

