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

#include <cuda_runtime_api.h>


template<typename T>
__device__ inline T __globalLd(T *addr) { return *addr; }
template<typename T>
__device__ inline T __globalLd(const T *addr) { return *addr; }


typedef unsigned char uint8_t;
#if defined(__GNUC__)
typedef unsigned long long_t;
#elif defined(_MSC_VER)
typedef ULONG_PTR long_t;
#endif

#define USE_VM_LEVELS 3

#define HARCODED

#ifdef HARCODED
#define __gmac_vm_shift_l1 39
#define __gmac_vm_shift_l2 33
#define __gmac_vm_shift_l3 21
#define __gmac_vm_mask_l1 (long_t(0))
#define __gmac_vm_mask_l2 (long_t(128  - 1) << 33)
#define __gmac_vm_mask_l3 (long_t(4096 - 1) << 21)

#else // HARCODED
__constant__ unsigned __gmac_vm_shift_l1;
__constant__ unsigned __gmac_vm_shift_l2;
__constant__ long_t __gmac_vm_mask_l1;
__constant__ long_t __gmac_vm_mask_l2;

#if USE_VM_LEVELS == 3
__constant__ unsigned __gmac_vm_shift_l3;
__constant__ long_t __gmac_vm_mask_l3;
#endif

#endif // HARCODED

#ifdef BITMAP_BYTE

#if 0
template<typename T>
__device__ __inline__ T __globalSt2(T *addr, T v) {
    *addr = v;
    uint8_t *walk = __gmac_vm_root[(addr & __gmac_vm_mask_l1) >> __gmac_vm_shift_l1];
    walk[(addr & __gmac_vm_mask_l2) >> __gmac_vm_shift_l2] = 1;
    return v;
}
#endif

#if USE_VM_LEVELS == 3
__constant__ uint8_t **__gmac_vm_root[1];

template<typename T>
__device__ __inline__ T __globalSt(T *addr, T v) {
    uint8_t &entry = __gmac_vm_root[0]
                                   [(long_t(addr) & __gmac_vm_mask_l2) >> __gmac_vm_shift_l2]
                                   [(long_t(addr) & __gmac_vm_mask_l3) >> __gmac_vm_shift_l3];
    *addr = v;
    entry = 1;
    return v;
}


#endif

#if USE_VM_LEVELS == 2
__constant__ uint8_t *__gmac_vm_root[1];

template<typename T>
__device__ __inline__ T __globalSt(T *addr, T v) {
    long_t index1 = (long_t(addr) & __gmac_vm_mask_l1) >> __gmac_vm_shift_l1;
    long_t index2 = (long_t(addr) & __gmac_vm_mask_l2) >> __gmac_vm_shift_l2;
    __gmac_vm_root[index1][index2] = 1;
    *addr = v;
    return v;
}
#endif

#if 0
#if USE_VM_LEVELS == 2
#define __globalSt(a,v) __globalSt2(a,v)
#else
#if USE_VM_LEVELS == 3
#define __globalSt(a,v) __globalSt3(a,v)
#else
#error "Unknown number of levels"
#endif
#endif
#endif


#else
#ifdef BITMAP_BIT
#define MASK_BITPOS ((1 << 5) - 1)
#define SHIFT_ENTRY 

__constant__ uint32_t *__gmac_vm_root;

template<typename T>
__device__ __inline__ T __globalSt(T *addr, T v) {
    *addr = v;
    long_t entry = to32bit(addr) >> (__SHIFT_PAGE + 5);
    uint32_t val = 1 << ((to32bit(addr) >> __SHIFT_PAGE) & MASK_BITPOS);
    atomicOr(&__gmac_vm_root[entry], val);
    return v;
}
#else
#define __globalSt(a,v) (*(a)=(v))
#endif
#endif


#endif
