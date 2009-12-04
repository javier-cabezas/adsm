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

#ifndef __MEMORY_VM_H_
#define __MEMORY_VM_H_

#include <config.h>
#include <threads.h>
#include <paraver.h>
#include <debug.h>

#include <stdint.h>
#include <malloc.h>
#include <math.h>

#include <cassert>

// Compiler check to ensure that defines set by configure script
// are consistent
#ifdef USE_VM_DEVICE
#ifndef USE_VM
#error "Virtual memory MUST be enabled to use device VM"
#endif
#endif

namespace gmac { namespace memory  { namespace vm {

typedef unsigned long addr_t;

//! This class is a hack to allow accessing a Context from
//! a Table, which is a template
class Dumper {
protected:
	void *alloc(size_t) const;
	void *hostAlloc(void **, size_t) const;

	void free(void *) const;
	void hostFree(void *) const;

	void flush(void *, const void *, size_t) const;
	void sync(void *, const void *, size_t) const;
};

template<typename T>
class Table : public Dumper {
protected:
	static const size_t defaultSize = 512;
	size_t nEntries;

	static const addr_t Present = 0x01;
	static const addr_t Dirty = 0x02;
	static const addr_t Mask = ~0x03;

	T **table;

	T *entry(size_t n) const {
		assert(n < nEntries);
		return (T *)((addr_t)table[n] & Mask);
	}

#ifdef USE_VM
	bool __shared;
	T **__shadow;
	void *__device;
#endif

public:
	Table(size_t nEntries = defaultSize) :
		nEntries(nEntries)
#ifdef USE_VM
		, __shared(true)
#endif
	{
		TRACE("Creating Table with %d entries (%p)", nEntries, this);

		assert(posix_memalign((void **)&table, 0x1000,
			nEntries * sizeof(T *)) == 0);
		memset(table, 0, nEntries * sizeof(T *));
#ifdef USE_VM
#ifdef USE_VM_DEVICE
		assert(posix_memalign((void **)&__shadow, 0x1000,
			nEntries * sizeof(T *)) == 0);
		__device = Dumper::alloc(nEntries * sizeof(T *));
#else
		__device = Dumper::hostAlloc((void **)&__shadow, nEntries * sizeof(T *));
#endif
		if(__shadow != NULL) memset(__shadow, 0, nEntries * sizeof(T *));
#endif
	}

	virtual ~Table()
	{
		TRACE("Cleaning Table with %d entries (%p)", nEntries, this);
		free(table);
#ifdef USE_VM
		enterFunction(vmFree);
#ifdef USE_VM_DEVICE
		free(__shadow);
		if(__device != NULL) Dumper::free(__device);
#else
		if(__shadow != NULL) Dumper::hostFree(__shadow);
#endif
		exitFunction();
#endif
	}

	inline bool present(size_t n) const {
		assert(n < nEntries);
		return (addr_t)table[n] & Present;
	}

	inline bool dirty(size_t n) const {
		assert(n < nEntries);
		return (addr_t)table[n] & Dirty;
	}

	inline void clean(size_t n) {
		assert(n < nEntries);
		table[n] = (addr_t *)((addr_t)table[n] & ~Dirty);
	}

#ifdef USE_VM
	void *device() { return __device; }
#endif

	void create(size_t n, size_t size = defaultSize) {
		enterFunction(vmAlloc);
		assert(n < nEntries);
		table[n] = (T *)((addr_t)new T(size) | Present);
#ifdef USE_VM
		__shared = false;
		__shadow[n] = (T *)((addr_t)entry(n)->device() | Present);
#endif
		exitFunction();
	}
	void insert(size_t n, void *addr) {
		assert(n < nEntries);
		table[n] = (T *)((addr_t)addr | Present);
#ifdef USE_VM
		__shadow[n] = (T *)((addr_t)addr | Present);
#endif
	}
	void remove(size_t n) {
		assert(n < nEntries);
		table[n] = (T *)0;
#ifdef USE_VM
		__shadow[n] = (T *)0;
#endif

	}
	inline T &get(size_t n) const { return *entry(n); }
	inline T *value(size_t n) const { return entry(n); }

	inline size_t size() const { return nEntries; }

	inline void realloc() {
#ifdef USE_VM
#ifdef USE_VM_DEVICE
		if(__device != NULL) Dumper::free(__device);
		__device = Dumper::alloc(nEntries * sizeof(T *));
#else
		if(__device != NULL) Dumper::hostFree(__shadow);
		__device = Dumper::hostAlloc((void **)&__shadow, nEntries * sizeof(T *));
		memset(__shadow, 0, nEntries * sizeof(T *));
#endif
		assert(__device != NULL);
#endif
	}

	inline void flush() const {
#ifdef USE_VM
		assert(__device != NULL);
		Dumper::flush(__device, __shadow, nEntries * sizeof(T *));
#endif
	}

	inline void sync() {
#ifdef USE_VM
		assert(__device != NULL);
		if(__shared == false) return;
		Dumper::sync(table, __device, nEntries * sizeof(T *));
#endif
	}
};



}}};

#endif
