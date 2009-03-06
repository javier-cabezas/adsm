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

# Get the NVIDIA CUDA compiler
AC_PATH_PROG([NVCC], [nvcc])
if test -z "$NVCC"; then
	AC_MSG_ERROR([CUDA compiler not found])
fi
CUDA_DIR="`dirname $NVCC`/.."

# Check for profiling mode
AC_MSG_CHECKING(whether to enable profiling mode)
profil_default="no"
AC_ARG_ENABLE(profile,
	AC_HELP_STRING([--enable-profile], [turn on profile mode [[default=no]]])
		, , enable_profile=$profile_default)
if test "$enable_profile" = "yes"; then
	CXXFLAGS="$CXXFLAGS -pg"
	LDFLAGS="$LDFLAGS -pg"
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi


# Check for debug mode
AC_MSG_CHECKING(whether to enable debug mode)
debug_default="no"
AC_ARG_ENABLE(debug,
	AC_HELP_STRING([--enable-debug], [turn on debug mode [[default=no]]])
		, , enable_debug=$debug_default)
if test "$enable_debug" = "yes"; then
	CXXFLAGS="$CXXFLAGS -O0 -fno-inline"
	CPPFLAGS="$CPPFLAGS -DDEBUG=1"
	AC_MSG_RESULT(yes)
else
	CXXFLAGS="-O3 -fomit-frame-pointer"
	LDGLAGS=""
	AC_MSG_RESULT(no)
fi


AC_SUBST(NVCC)
AC_SUBST(CUDA_DIR)

AC_OUTPUT(
	Makefile
	src/Makefile
	src/common/Makefile
	src/common/config.h
	src/lib/Makefile
	src/inter/Makefile
	tests/Makefile
)
