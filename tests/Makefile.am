MAINTAINERCLEANFILES=Makefile.in

bin_PROGRAMS = vecAdd gmacVecAdd

vecAdd_SOURCES = vecAdd.cu

gmacVecAdd_SOURCES = gmacVecAdd.cu
gmacVecAdd_CPPFLAGS = -I$(top_srcdir)/src/lib
gmacVecAdd_LDADD = ../src/lib/.libs/libgmac.a

export LDFLAGS = -L../src/lib/.libs
LINK = $(NVCC) -o $@ $(LDFLAGS) $(LDADD)

.cu.o:
	$(NVCC) -c $< -o $@ $($(basename $@)_CPPFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS)

