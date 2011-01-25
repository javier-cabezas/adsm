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

#ifndef GMAC_MEMORY_MEMORY_H_
#define GMAC_MEMORY_MEMORY_H_

#include "config/common.h"
#include "include/gmac/types.h"

#include "util/Logger.h"

namespace __impl { namespace memory {

extern size_t BlockSize_;
#ifdef USE_VM
extern size_t SubBlockSize_;
extern unsigned BlockShift_;
extern unsigned SubBlockShift_;
extern long_t SubBlockMask_;
#endif

class GMAC_LOCAL Memory {
public:
	static int protect(hostptr_t addr, size_t count, GmacProtection prot);
	static hostptr_t map(hostptr_t addr, size_t count, GmacProtection prot = GMAC_PROT_NONE);
	static hostptr_t shadow(hostptr_t addr, size_t count);
	static void unshadow(hostptr_t addr, size_t count);
	static void unmap(hostptr_t addr, size_t count);
};

#if USE_VM

static
long_t log2(long_t n)
{
    long_t ret = 0;
    while (n != 1) {
        ASSERTION((n & 0x1) == 0);
        n >>= 1;
        ret++;
    }
    return ret;
}

static
unsigned log2(unsigned n)
{
    unsigned ret = 0;
    while (n != 1) {
        ASSERTION((n & 0x1) == 0);
        n >>= 1;
        ret++;
    }
    return ret;
}

static inline
long_t
GetSubBlock(const hostptr_t _addr)
{
    long_t addr = long_t(_addr);
    return (addr >> SubBlockShift_) & SubBlockMask_;
}

static inline
hostptr_t
GetBlockAddr(const hostptr_t _start, const hostptr_t _addr)
{
    long_t start = long_t(_start);
    long_t addr = long_t(_addr);
    long_t off = addr - start;
    long_t block = off / BlockSize_;
    return hostptr_t(start + block * BlockShift_);
}

static inline
hostptr_t
GetSubBlockAddr(const hostptr_t _start, const hostptr_t _addr)
{
    long_t start = long_t(_start);
    long_t addr  = long_t(_addr);
    long_t off = addr - start;
    long_t subBlock = off / SubBlockSize_;
    return hostptr_t(start + subBlock * SubBlockShift_);
}
#endif

}}

#endif
