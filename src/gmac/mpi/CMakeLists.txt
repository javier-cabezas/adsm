
include_directories(${KDE4_INCLUDES} ${KDE4_INCLUDE_DIR} ${QT_INCLUDES} )


########### next target ###############

set(gmac-mpi_LIB_SRCS ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)

kde4_add_library(gmac-mpi SHARED ${gmac-mpi_LIB_SRCS})

target_link_libraries(gmac-mpi ${KDE4_KDECORE_LIBS})

set_target_properties(gmac-mpi PROPERTIES VERSION 1.0.0 SOVERSION 1)
install(TARGETS gmac-mpi ${INSTALL_TARGETS_DEFAULT_ARGS})


########### install files ###############




#original Makefile.am contents follow:

#MAINTAINERCLEANFILES = Makefile.in
#
#CXX=mpic++
#LD=mpic++
#
#lib_LTLIBRARIES = libgmac-mpi.la
## libgmac_stdc_la_SOURCES = stdc.h stdc.cpp mem.cpp io.cpp 
#libgmac_mpi_la_SOURCES = mpi_local.h mpi.cpp
#libgmac_mpi_la_LIBADD = -lgmac
