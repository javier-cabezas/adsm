# Introduction #

Compilation for 32-bit MacOS is not straightforward due to the old GNU tools included in XCode, and a number of bugs in recent versions of GNU libtool for this platform.

**IMPORTANT: this procedure is only necessary in 32-bit MacOS X. If you have a 64-bit machine, follow the standard compilation procedure.**

The procedure here explained can be applied to both the source code archive and to a fresh checkout from the project code repository.


# The easy way #

> You need the following versions of GNU tools installed:
    * GCC 4.0
    * GNU Libtool 2.2.10
You can check your versions numbers as follows:
```
igelado@idefix:~/$ gcc-4.0 --version
i686-apple-darwin10-gcc-4.0.1 (GCC) 4.0.1 (Apple Inc. build 5494)
Copyright (C) 2005 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

igelado@idefix:~/$ glibtool --version
libtool (GNU libtool) 2.2.10
Written by Gordon Matzigkeit <gord@gnu.ai.mit.edu>, 1996

Copyright (C) 2010 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
You might want to install them using [MacPorts](http://www.macports.org).

If you have this versions, then compiling GMAC is as easy as:
```
igelado@idefix:~/$CC=gcc-4.0 CXX=g++-4.0 ./configure --disable-mmap
```

# The hard way #

If you have a higher version of GCC and do not want to install version 4.0, you can also try modifying the file `config.guess` in the root directory of GMAC code. You need to look for some lines similar to:
```
    UNAME_PROCESSOR=`uname -p` || UNAME_PROCESSOR=unknown
    case $UNAME_PROCESSOR in
        i386)
        eval $set_cc_for_build
        if [ "$CC_FOR_BUILD" != 'no_compiler_found' ]; then
          if (echo '#ifdef __LP64__'; echo IS_64BIT_ARCH; echo '#endif') | \
              (CCOPTS= $CC_FOR_BUILD -E - 2>/dev/null) | \
              grep IS_64BIT_ARCH >/dev/null
          then
              UNAME_PROCESSOR="amd64"
          fi
        fi ;;
        unknown) UNAME_PROCESSOR=powerpc ;;
    esac

```
This piece of code is trying to (incorrectly) detect your processor architecture. Modify the assignment of `UNAME_PROCESSOR="amd64"` to `UNAME_PROCESSOR="i386"`. Note that this hack is required anytime the `config.guess` file is modified (e.g., whenever you run the `setup.sh` script).