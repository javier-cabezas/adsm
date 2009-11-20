AC_INIT(GMAC, 2.0, igelado@ac.upc.edu, gmac)
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CONFIG_MACRO_DIR([m4])
AC_GNU_SOURCE

AC_PROG_INSTALL
AC_PROG_CC
AC_PROG_CXX
AC_PROG_LD
AUTOCONF_ENV="SHELL=/bin/sh"
AM_PROG_LIBTOOL

# Check for a hash table implementation
AC_CXX_HEADER_TR1_UNORDERED_MAP

# Check for the operating system
case "$host" in
*linux* | *Linux*)
AC_MSG_RESULT([Linux])
OS_DIR="posix"
;;
*)
AC_MSG_RESULT([Unknown])
AC_MSG_ERROR([Your operating system is not supported])
;;
esac

# Check for pthread library
AC_CHECK_LIB([pthread], [pthread_create],
	[AC_DEFINE([HAVE_LIBPTHREAD], [1])
	LIBS="-lpthread $LIBS"; thread_name="pthreads"]
)

# Get the NVIDIA CUDA compiler
AC_PATH_PROG([NVCC], [nvcc])
if test -z "$NVCC"; then
	AC_MSG_ERROR([CUDA compiler not found])
fi
CUDA_DIR="`dirname $NVCC`/.."
API_DIR="cuda"
AC_DEFINE([HAVE_CUDA], [1])

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


# Check for paraver tracing
#"
AC_MSG_CHECKING(whether to enable paraver tracing)
paraver_default="no"
AC_ARG_ENABLE(paraver,
	AC_HELP_STRING([--enable-paraver], [turn on paraver tracing [[default=no]]])
		, , enable_paraver=$paraver_default)
if test "$enable_paraver" = "yes"; then
	AC_MSG_RESULT(yes)
    AC_CHECK_LIB([paraver], [paraver_lib_present],
                 [LIBS="-lparaver $LIBS"] [paraver_lib=""],
                 [paraver_lib="\$(abs_top_builddir)/lib/libparaver/paraver/libparaver.la"]
                 [paraver_include="-I\$(abs_top_srcdir)/lib/libparaver -I\$(abs_top_builddir)/lib/libparaver"]
                 [paraver_src="libparaver"]
                 [AC_CONFIG_SUBDIRS([lib/libparaver])]
                 )

	AC_DEFINE(PARAVER)
else
	paraver_lib=""
	AC_MSG_RESULT(no)
fi


# Check for driver mode
cuda_driver_default="no"
AC_MSG_CHECKING(whether to use CUDA Driver API)
AC_ARG_ENABLE(cuda-driver,
	AC_HELP_STRING([--enable-cuda-driver], [turn on using the CUDA driver API [[default=no]]])
		, , enable_cuda_driver=$cuda_driver_default)
if test "$enable_cuda_driver" = "yes"; then
	API_DIR="cudadrv"
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi

# Check for software virtual memory support
vm_host_default="no"
AC_MSG_CHECKING(whether to use host memory to store VM data structures)
AC_ARG_ENABLE(vm-host,
	AC_HELP_STRING([--enable-vm-host], [turn on using host memory to store VM data structures [[default=no]]])
		, , enable_host_vm=$vm_host_default)
if test "$enable_vm_host" = "yes"; then
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi

vm_default="no"
AC_MSG_CHECKING(whether to use software virtual memory)
AC_ARG_ENABLE(vm,
	AC_HELP_STRING([--enable-vm], [turn on software virtual memory on CUDA [[default=no]]])
		, , enable_vm=$vm_default)
if test "$enable_vm" = "yes" || test "$enable_vm_host" = "yes"; then
	AC_DEFINE([USE_VM])
	if test "$enable_host_vm" = "no"; then
		AC_DEFINE([USE_VM_DEVICE])
	fi
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi

# Check for using map to request host memory
mmap_default="no"
AC_MSG_CHECKING(whether to use mmap() to allocate system memory)
AC_ARG_ENABLE(mmap,
	AC_HELP_STRING([--enable-mmap], [turn on using mmap() to request system memory [[default=no]]])
		, , enable_mmap=$mmap_default)
if test "$enable_mmap" = "yes"; then
	if test "$enable_vm" = "yes"; then
		AC_MSG_ERROR("mmap() cannot be used when using software virtual memory")
	fi
	AC_DEFINE([USE_MMAP])
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi

global_host_default="no"
AC_MSG_CHECKING(whether to use a host memory for global allocations)
AC_ARG_ENABLE(global-host,
	AC_HELP_STRING([--enable-global-host], [turn on host memory for global allocations [[default=no]]])
		, , enable_global_host=$global_host_default)
if test "$enable_global_host" = "yes"; then
	AC_DEFINE([USE_GLOBAL_HOST])
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi

async_launch_default="no"
AC_MSG_CHECKING(whether to perform asynchronous kernel launches using streams)
AC_ARG_ENABLE(async-launch,
    AC_HELP_STRING([--enable-async-launch], [turn on asynchronous kernel launches [[default=no]]])
        , , enable_async_launch=$async_launch_default)
if test "$enable_async_launch" = "yes"; then
    AC_DEFINE([USE_ASYNC_LAUNCH])
    AC_MSG_RESULT(yes)
else
    AC_MSG_RESULT(no)
fi

AC_SUBST(NVCC)
AC_SUBST(CUDA_DIR)
AC_SUBST(API_DIR)
AC_SUBST(OS_DIR)
AC_SUBST(THREAD_DIR, $thread_name)

AC_SUBST(PARAVER_INCLUDE, $paraver_include)
AC_SUBST(PARAVER_LIB, $paraver_lib)
AC_SUBST(PARAVER_SRC, $paraver_src)

AC_CONFIG_LINKS(src/config/threads.h:src/config/$thread_name.h)

AC_OUTPUT(
	Makefile
	lib/Makefile
	src/Makefile
	src/config/config.h
	src/kernel/Makefile
	src/memory/Makefile
	src/memory/posix/Makefile
	src/gmac/Makefile
	src/gmac/stdc/Makefile
	src/gmac/posix/Makefile
	src/gmac/pthreads/Makefile
	src/api/Makefile
	src/api/cuda/Makefile
	src/api/cudadrv/Makefile
	src/os/Makefile
	src/util/Makefile
	tests/Makefile
)
