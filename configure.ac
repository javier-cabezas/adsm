AC_INIT([GMAC],[3.0],[igelado@ac.upc.edu],[gmac])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CONFIG_MACRO_DIR([m4])
AC_USE_SYSTEM_EXTENSIONS

AC_PROG_INSTALL
AC_PROG_CC
AC_PROG_CXX
LT_PATH_LD
AUTOCONF_ENV="SHELL=/bin/sh"
LT_INIT

# Check for a hash table implementation
AC_CXX_HEADER_TR1_UNORDERED_MAP

# Check for processor
AC_MSG_CHECKING(processor architecture)
case "$host" in
*x86_64*)
AC_MSG_RESULT([amd64])
ARCH="64"
;;
*i386*|*i686*)
AC_MSG_RESULT([ia32])
ARCH="32"
;;
*)
AC_MSG_RESULT([Unknown: "$host"])
AC_MSG_ERROR([Your processor is not supported])
;;
esac


# Check for the operating system
case "$host" in
*linux* | *Linux*)
AC_MSG_RESULT([Linux])
OS_DIR="posix"
AC_DEFINE(LINUX)
LN_FLAGS=-Tf
;;
*apple* | *Apple*)
AC_MSG_RESULT([MacOS])
OS_DIR="posix"
AC_DEFINE(DARWIN)
LN_FLAGS=-f
;;
*)
AC_MSG_RESULT([Unknown: "$host"])
AC_MSG_ERROR([Your operating system is not supported])
;;
esac

# Check for functions we need
AC_CHECK_FUNCS(posix_memalign)

# Check for pthread library
AC_CHECK_LIB([pthread], [pthread_create],
	[AC_DEFINE([HAVE_LIBPTHREAD], [1])
	thread_name="pthreads"]
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
	AS_HELP_STRING([--enable-debug],[turn on debug mode [[default=no]]])
		, , enable_debug=$debug_default)
if test "$enable_debug" = "yes"; then
	AC_MSG_RESULT(yes)
	CFLAGS="-Wall -Wno-unused-variable -Werror -g -O0"
	CXXFLAGS="-Wall -Wno-unused-variable -Werror -g -O0 -fno-inline"
    CUDA_CFLAGS="-g -O0"
	AC_DEFINE(DEBUG)
else
	AC_MSG_RESULT(no)
	CXXFLAGS="-Wall -Wno-unused-variable -Werror -g -O3 -fomit-frame-pointer"
	CFLAGS="-Wall -Wno-unused-variable -Werror -g -O3 -fomit-frame-pointer"
    CUDA_CFLAGS="-g -O3 -fomit-frame-pointer"
fi

# Check for profiling mode
AC_MSG_CHECKING(whether to enable profiling mode)
profile_default="no"
AC_ARG_ENABLE(profile,
	AS_HELP_STRING([--enable-profile],[turn on profile mode [[default=no]]])
		, , enable_profile=$profile_default)
if test "$enable_profile" = "yes"; then
	AC_MSG_RESULT(yes)
	CFLAGS="$( echo $CFLAGS | sed s/-fomit-frame-pointer//g )"
	CXXFLAGS="$( echo $CXXFLAGS | sed s/-fomit-frame-pointer//g )"
	CXXFLAGS="$CXXFLAGS -pg"
	LDFLAGS="$LDFLAGS -pg"
else
	AC_MSG_RESULT(no)
fi


# Check for paraver tracing
#"
AC_MSG_CHECKING(whether to enable paraver tracing)
paraver_default="no"
AC_ARG_ENABLE(paraver,
	AS_HELP_STRING([--enable-paraver],[turn on paraver tracing [[default=no]]])
		, , enable_paraver=$paraver_default)
if test "$enable_paraver" = "yes"; then
	AC_MSG_RESULT(yes)
   AC_MSG_CHECKING(whether to compile libparaver from lib directory)
   AC_ARG_ENABLE(compile-paraver,
      AC_HELP_STRING([--enable-compile-paraver], [force compiling libparaver from lib directory [[default=no]]]),
      [AC_MSG_RESULT(yes)]
      [paraver_lib="\$(abs_top_builddir)/lib/libparaver/paraver/libparaver.la"]
      [paraver_include="-I\$(abs_top_srcdir)/lib/libparaver -I\$(abs_top_builddir)/lib/libparaver"]
      [paraver_src="libparaver"],
      [AC_MSG_RESULT(no)]
      [AC_CHECK_LIB([paraver], [paraver_lib_present],
         [paraver_lib="-lparaver"],
         [paraver_include=""]
         [paraver_src=""]
      )]
   )

   if test "$paraver_lib" != ""; then
      AC_CONFIG_SUBDIRS([lib/libparaver])
   fi 

	AC_DEFINE(PARAVER)
else
	paraver_lib=""
	AC_MSG_RESULT(no)
fi


# Check for driver mode
cuda_driver_default="no"
AC_MSG_CHECKING(whether to use CUDA Driver API)
AC_ARG_ENABLE(cuda-driver,
	AS_HELP_STRING([--enable-cuda-driver],[turn on using the CUDA driver API [[default=no]]])
		, , enable_cuda_driver=$cuda_driver_default)
if test "$enable_cuda_driver" = "yes"; then
	AC_MSG_RESULT(yes)
	API_DIR="cudadrv"
else
	AC_MSG_RESULT(no)
fi

# Check for software virtual memory support
vm_host_default="no"
AC_MSG_CHECKING(whether to use host memory to store VM data structures)
AC_ARG_ENABLE(vm-host,
	AS_HELP_STRING([--enable-vm-host],[turn on using host memory to store VM data structures [[default=no]]])
		, , enable_host_vm=$vm_host_default)
if test "$enable_vm_host" = "yes"; then
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi

vm_default="no"
AC_MSG_CHECKING(whether to use software virtual memory)
AC_ARG_ENABLE(vm,
	AS_HELP_STRING([--enable-vm],[turn on software virtual memory on CUDA [[default=no]]])
		, , enable_vm=$vm_default)
if test "$enable_vm" = "yes" || test "$enable_vm_host" = "yes"; then
	AC_MSG_RESULT(yes)
	AC_DEFINE([USE_VM])
	if test "$enable_host_vm" = "no"; then
		AC_DEFINE([USE_VM_DEVICE])
	fi
else
	AC_MSG_RESULT(no)
fi

# Check for using map to request host memory
mmap_default="no"
AC_MSG_CHECKING(whether to use mmap() to allocate system memory)
AC_ARG_ENABLE(mmap,
	AS_HELP_STRING([--enable-mmap],[turn on using mmap() to request system memory [[default=no]]])
		, , enable_mmap=$mmap_default)
if test "$enable_mmap" = "yes"; then
	AC_MSG_RESULT(yes)
	if test "$enable_vm" = "yes"; then
		AC_MSG_ERROR("mmap cannot be used when using software virtual memory")
	fi
   if test "$arch" = "32"; then
      AC_MSG_ERROR("mmap cannot be used in 32bit architectures")
   fi
	AC_DEFINE([USE_MMAP])
else
	AC_MSG_RESULT(no)
fi

global_host_default="no"
AC_MSG_CHECKING(whether to use a host memory for global allocations)
AC_ARG_ENABLE(global-host,
	AS_HELP_STRING([--enable-global-host],[turn on host memory for global allocations [[default=no]]])
		, , enable_global_host=$global_host_default)
if test "$enable_global_host" = "yes"; then
    if test "$enable_mmap" = "yes"; then
        AC_MSG_ERROR("mmap() cannot be used when host-mapped global regions are used")
    fi
	AC_MSG_RESULT(yes)
	AC_DEFINE([USE_GLOBAL_HOST])
else
	AC_MSG_RESULT(no)
fi

multi_context_default="no"
AC_MSG_CHECKING(whether to use multiple device contexts per device)
AC_ARG_ENABLE(multi-context,
	AS_HELP_STRING([--enable-multi-context],[turn on multiple context per accelerator on [[default=no]]])
		, , enable_multi_context=$multi_context_default)
if test "$enable_multi_context" = "yes" || test "$enable_multi_context" = "yes"; then
	AC_MSG_RESULT(yes)
	AC_DEFINE([USE_MULTI_CONTEXT])
else
	AC_MSG_RESULT(no)
fi

tests_default="no"
AC_MSG_CHECKING(whether to build tests)
AC_ARG_ENABLE(tests,
    AS_HELP_STRING([--enable-tests],[turn on tests [[default=no]]])
        , , enable_tests=$tests_default)
if test "$enable_tests" = "yes"; then
    AC_MSG_RESULT(yes)
    BUILD_TESTS="tests"
else
    BUILD_TESTS=""
    AC_MSG_RESULT(no)
fi

AC_SUBST(ARCH)
AC_SUBST(BUILD_TESTS)
AC_SUBST(NVCC)
AC_SUBST(CUDA_DIR)
AC_SUBST(API_DIR)
AC_SUBST(OS_DIR)
AC_SUBST(LN_FLAGS)
AC_SUBST(THREAD_DIR, $thread_name)

AC_SUBST(PARAVER_INCLUDE, $paraver_include)
AC_SUBST(PARAVER_LIB, $paraver_lib)
AC_SUBST(PARAVER_SRC, $paraver_src)

AC_CONFIG_LINKS(src/config/threads.h:src/config/$thread_name.h)

AC_CONFIG_FILES([Makefile
	lib/Makefile
    doc/Doxyfile
    doc/Makefile
	src/Makefile
	src/config/config.h
	src/config/Makefile
	src/kernel/Makefile
	src/memory/Makefile
	src/memory/manager/Makefile
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
])
AC_OUTPUT
