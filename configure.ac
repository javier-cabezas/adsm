AC_INIT(GMAC, 2.0, igelado@ac.upc.edu, gmac)
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_GNU_SOURCE

AC_PROG_INSTALL
AC_PROG_CC
AC_PROG_CXX
AC_PROG_LD
AUTOCONF_ENV="SHELL=/bin/sh"
AM_PROG_LIBTOOL

# Check for a hash table implementation
AC_CXX_HEADER_TR1_UNORDERED_MAP

# Check for pthread library
AC_CHECK_LIB([pthread], [pthread_create],
	[AC_DEFINE([HAVE_LIBPTHREAD], [1])
	LIBS="-lpthread $LIBS"; thread_dir="pthread"]
)

# Get the NVIDIA CUDA compiler
AC_PATH_PROG([NVCC], [nvcc])
if test -z "$NVCC"; then
	AC_MSG_ERROR([CUDA compiler not found])
fi
CUDA_DIR="`dirname $NVCC`/.."

# Check for debug mode - MUST BE THE FIRST CHECK
AC_MSG_CHECKING(whether to enable debug mode)
debug_default="no"
AC_ARG_ENABLE(debug,
	AC_HELP_STRING([--enable-debug], [turn on debug mode [[default=no]]])
		, , enable_debug=$debug_default)
if test "$enable_debug" = "yes"; then
	CFLAGS="-g -O0"
	CXXFLAGS="-g -O0 -fno-inline"
	AC_DEFINE(DEBUG)
	AC_MSG_RESULT(yes)
else
	CXXFLAGS="-O3 -fomit-frame-pointer"
	CFLAGS="-O3"
	AC_MSG_RESULT(no)
fi

# Check for profiling mode
AC_MSG_CHECKING(whether to enable profiling mode)
profile_default="no"
AC_ARG_ENABLE(profile,
	AC_HELP_STRING([--enable-profile], [turn on profile mode [[default=no]]])
		, , enable_profile=$profile_default)
if test "$enable_profile" = "yes"; then
	CFLAGS="$( echo $CFLAGS | sed s/-fomit-frame-pointer//g )"
	CXXFLAGS="$( echo $CXXFLAGS | sed s/-fomit-frame-pointer//g )"
	CXXFLAGS="$CXXFLAGS -pg"
	LDFLAGS="$LDFLAGS -pg"
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi


# Check for paraver tracing on library
paraver_default="no"
paraver_gmac_default="no"

AC_MSG_CHECKING(whether to enable paraver tracing of library)
AC_ARG_ENABLE(paraver-gmac,
	AC_HELP_STRING([--enable-paraver-gmac], [turn no paraver tracing on library [[defalult=no]]])
		, , enable_paraver_gmac=$paraver_gmac_default)
if test "$enable_paraver_gmac" = "yes"; then
	paraver_default="yes"
	AC_DEFINE(PARAVER_GMAC)
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi


# Check for paraver tracing
AC_MSG_CHECKING(whether to enable paraver tracing)
AC_ARG_ENABLE(paraver,
	AC_HELP_STRING([--enable-paraver], [turn on paraver tracing [[default=no]]])
		, , enable_paraver=$paraver_default)
if test "$enable_paraver" = "yes"; then
	paraver_dir="paraver"
	paraver_lib="\$(top_builddir)/src/paraver/libparaver.la"
	AC_DEFINE(PARAVER)
	AC_MSG_RESULT(yes)
else
	paraver_dir=""
	paraver_dir=""
	AC_MSG_RESULT(no)
fi



AC_SUBST(NVCC)
AC_SUBST(CUDA_DIR)
AC_SUBST(THREAD_DIR, $thread_dir)

AC_SUBST(PARAVER_DIR, $paraver_dir)
AC_SUBST(PARAVER_LIB, $paraver_lib)

AC_OUTPUT(
	Makefile
	src/Makefile
	src/common/Makefile
	src/common/config.h
	src/common/unix/Makefile
	src/paraver/Makefile
	src/paraver/config.h
	src/paraver/linux/Makefile
	src/lib/Makefile
	src/lib/unix/Makefile
	src/lib/pthread/Makefile
	src/inter/Makefile
	tests/Makefile
)
