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


#ifndef __GMAC_VM_H_
#define __GMAC_VM_H_


#define PRESENT 0x1
#define DIRTY 0x2
#define MASK 0x3

#define ENTRIES 512
#define ROOT_SHIFT 39
#define DIR_SHIFT 30

#define __entry(addr, shift, size) \
	((unsigned long)addr >> shift) & (size - 1)
#define __value(tbl, n) ((unsigned long)tbl[n] & ~MASK)

__constant__ struct {
	unsigned long **ptr;
	size_t shift;
	size_t size;
	size_t page;
} __pageTable;

__device__ inline void *__pg_lookup(const void *addr, bool write)
{
	unsigned long offset = (unsigned long)addr & (__pageTable.page - 1);
	unsigned long *dir = (unsigned long *)__value(__pageTable.ptr,
		__entry(addr, ROOT_SHIFT, ENTRIES));
	unsigned long *table = (unsigned long *)__value((unsigned long **)dir,
		__entry(addr, DIR_SHIFT, ENTRIES));
	unsigned long base = __value((unsigned long **)table, __entry(addr,
		__pageTable.shift, __pageTable.size));
	if(write)
		table[__entry(addr, __pageTable.shift, __pageTable.size)] |= DIRTY;
	base += offset;

	return (void *)base;
}

template<typename T>
__device__ inline T __globalLd(T *addr) { return *(T *)__pg_lookup(addr, false); }
template<typename T>
__device__ inline T __globalLd(const T *addr) { return *(const T *)__pg_lookup(addr, false); }

template<typename T>
__device__ inline void __globalSt(T *addr, T v) { *(T *)__pg_lookup(addr, true) = v; }


#endif
