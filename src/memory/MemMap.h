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

#ifndef __MEMORY_MAPMANAGER_H_
#define __MEMORY_MAPMANAGER_H_

#include <threads.h>

#include <assert.h>
#include <map>

namespace gmac {

template<typename T>
class MemMap {
protected:
	typedef std::map<const void *, T *> Map;
	Map __map;
	MUTEX(__mutex);
public:
	typedef typename Map::iterator iterator;
	typedef typename Map::const_iterator const_iterator;

	MemMap() { MUTEX_INIT(__mutex); }

	virtual ~MemMap() { 
		typename Map::iterator i;
		MUTEX_LOCK(__mutex);
		for(i = __map.begin(); i != __map.end(); i++) {
			delete i->second;
		}
		__map.clear();
		MUTEX_UNLOCK(__mutex);
	}

	inline void lock() { MUTEX_LOCK(__mutex); }
	inline void unlock() { MUTEX_UNLOCK(__mutex); }
	inline typename Map::iterator begin() { return __map.begin(); }
	inline typename Map::iterator end() { return __map.end(); }

	inline void insert(T *i) {
		void *key = (void *)((addr_t)i->getAddress() + i->getSize());
		MUTEX_LOCK(__mutex);
		__map.insert(typename Map::value_type(key, i));
		MUTEX_UNLOCK(__mutex);
	}

	inline T *remove(void *addr) {
		typename Map::iterator i;
		MUTEX_LOCK(__mutex);
		i = __map.upper_bound(addr);
		if(i == __map.end() || i->second->getAddress() != addr)
			FATAL("Bad free for %p", addr);
		T *ret = i->second;
		__map.erase(i);
		MUTEX_UNLOCK(__mutex);
		return ret;
	}

	virtual T *find(void *addr) {
		typename Map::const_iterator i;
		MUTEX_LOCK(__mutex);
		i = __map.upper_bound(addr);
		MUTEX_UNLOCK(__mutex);
		if(i == __map.end() || *(i->second) != addr) return NULL;
		return i->second;
	}

	// Gets the first memory region (CPU or accelerator) that
	// includes the memory range.
	// \param addr Starting address of the memory range
	// \param size Size (in bytes) of the memory range
	// \param Memory region where the range starts or NULL if
	// the range starts at CPU memory
	size_t filter(const void *addr, size_t size, MemRegion *&reg) {
		size_t ret = 0;
		typename Map::iterator i;
		MUTEX_LOCK(__mutex);
		i = __map.upper_bound(addr);
		// All the range owns to the CPU
		if(i == __map.end() || i->second->contains(addr, size) == false) {
			ret = size;
			reg = NULL;
		}
		// The range starts at the accelerator 
		else if((addr_t)addr >= (addr_t)i->second->getAddress()) {
			reg = i->second;
			ret = reg->getSize() - ((addr_t)addr - (addr_t)reg->getAddress());;
		}
		// The range starts at the CPU but includes accelerator memory
		else {
			ret = (addr_t)i->second->getAddress() - (addr_t)addr;
			reg = NULL;
		}
		MUTEX_UNLOCK(__mutex);
		return ret;
	}

};

};

#endif
