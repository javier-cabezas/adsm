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


#ifndef GMAC_USER_VM_H_
#define GMAC_USER_VM_H_

#include <stdint.h>

#include <cuda_runtime_api.h>


template<typename T>
__device__ inline T __globalLd(T *addr) { return *addr; }
template<typename T>
__device__ inline T __globalLd(const T *addr) { return *addr; }

__constant__ int __SHIFT_ENTRY;
__constant__ int __SHIFT_PAGE;

#define to32bit(a) ((unsigned long)a & 0xffffffff)

#ifdef BITMAP_WORD
__constant__ uint32_t *__dirtyBitmap;

template<typename T>
__device__ __inline__ void __globalSt(T *addr, T v) {
    *addr = v;
    __dirtyBitmap[to32bit(addr) >> __SHIFT_PAGE] = 1;
}
#else
#ifdef BITMAP_BYTE
__constant__ uint8_t *__dirtyBitmap;

template<typename T>
__device__ __inline__ void __globalSt(T *addr, T v) {
    *addr = v;
    __dirtyBitmap[to32bit(addr) >> __SHIFT_PAGE] = 1;
}
#else
#ifdef BITMAP_BIT
#define __SHIFT_BITPOS  (__SHIFT_ENTRY + 5)
#define MASK_BITPOS  ((1 << 5) - 1)

__constant__ uint32_t *__dirtyBitmap;

template<typename T>
__device__ __inline__ void __globalSt(T *addr, T v) {
    *addr = v;
    unsigned long entry = to32bit(addr) >> __SHIFT_BITPOS;
    uint32_t val = 1 << ((to32bit(addr) >> __SHIFT_ENTRY) & MASK_BITPOS);
    atomicOr(&__dirtyBitmap[entry], val);
}
#else
//#error "Bitmap granularity not defined"
#define __globalSt(a,v) *(a)=(v)
#endif
#endif
#endif


#endif
