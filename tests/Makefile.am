MAINTAINERCLEANFILES=Makefile.in

bin_PROGRAMS = vecAdd shVecAdd

vecAdd_SOURCES = vecAdd.cu

shVecAdd_SOURCES = shVecAdd.cu
shVecAdd_CPPFLAGS = -I$(top_srcdir)/src/lib
shVecAdd_LDADD = -lshcuda

export LDFLAGS = -L../src/lib/.libs
LINK = $(NVCC) -o $@ $(LDFLAGS) $(LDADD)

.cu.o:
	$(NVCC) -c $< -o $@ $($(basename $@)_CPPFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS)

