MAINTAINERCLEANFILES=Makefile.in

bin_PROGRAMS = vecAdd gmacVecAdd gmacThreadVecAdd

vecAdd_SOURCES = vecAdd.cu
vecAdd_LDADD = -L$(CUDA_DIR)/lib -lcuda -lcudart

gmacVecAdd_SOURCES = gmacVecAdd.cu
gmacVecAdd_CPPFLAGS = -include $(top_srcdir)/src/lib/gmac/cuda.h -I$(top_srcdir)/src/lib
gmacVecAdd_LDADD = ../src/lib/libgmac.la 

gmacThreadVecAdd_SOURCES = gmacThreadVecAdd.cu
gmacThreadVecAdd_CPPFLAGS = -include $(top_srcdir)/src/lib/gmac/cuda.h -I$(top_srcdir)/src/lib
gmacThreadVecAdd_LDADD = ../src/lib/libgmac.la -lpthread


LINK = $(LIBTOOL) --mode=link $(CC) -static $(CFLAGS) -o $@ 

.cu.o:
	$(NVCC) -c $< -o $@ $($(basename $@)_CPPFLAGS) $(CFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS) $(CFLAGS)

