/* Copyright (c) 2009 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef __CONFIG_DEBUG_H
#define __CONFIG_DEBUG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define __THREAD_CANARY
#include <threads.h>
#undef __THREAD_CANARY


#if defined(__LP64__) 
#define FMT_TID "0x%lx"
#else
#if defined(DARWIN)
#define FMT_TID "%p"
#else
#define FMT_TID "0x%llx"
#endif
#endif

#ifdef DEBUG
#define ASSERT(e)  \
    if (!(e)) {    \
		fprintf(stderr,"ASSERT ERROR at [%s:%d]\n", __FILE__, __LINE__); \
        abort();   \
    }
#else
#define ASSERT(e)
#endif

#define FATAL(fmt, ...)	\
	do {	\
		fprintf(stderr,"FATAL [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);	\
		abort();	\
	} while(0)

#define CFATAL(e, fmt, ...) \
	do {	                \
        if (!(e)) {         \
		    fprintf(stderr,"FATAL [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);	\
		    abort();	    \
        }                   \
	} while(0)

#define CBREAK(e, ...) \
        if (!(e)) {      \
            __VA_ARGS__; \
		    break;	     \
        }

#ifdef DEBUG
#define TRACE(fmt, ...)	\
	do {	\
		fprintf(stderr,"TRACE [%s:%d] ("FMT_TID")" fmt "\n",  __FILE__, __LINE__, SELF(), ##__VA_ARGS__);	\
	} while(0)
#else
#define TRACE(fmt, ...)
#endif

#ifdef DEBUG
#define WARNING(fmt, ...) \
   do { \
      fprintf(stderr,"WARNING [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
   } while(0)
#else
#define WARNING(fmt, ...)
#endif

#endif
