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

#include <config/threads.h>

#include <assert.h>
#include <map>

namespace gmac {

template<typename T>
class MemMap {
protected:
	typedef std::map<void *, T *> Map;
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

	bool split(void *addr, size_t size, RegionList &cpu,
			RegionList &acc)
	{
		typename Map::iterator i;
		bool ret = false;
		MUTEX_LOCK(__mutex);
		i = __map.upper_bound(addr);
		// There is no region containing this memory interval
		if(i == __map.end() || i->second->contains(addr, size) == false) {
			cpu.push_back(MemRegion(addr, size));
		}
		// Check for a region having a perfect match (common case)
		else if(i->second->equals(addr, size)) {
			acc.push_back(MemRegion(addr, size));
			i->second->invalidate();
		}
		// The invalidation might affect one or more regions and
		// require partial invalidation
		else {
			ret = true;
			addr_t _addr = (addr_t) addr;
			size_t _size = size;
			while(i->second->contains(addr, size)) {
				addr_t _region = (addr_t) i->second->getAddress();
				// Check for a CPU range at the begining
				if(_addr < _region) {
					cpu.push_back(MemRegion((void *)_addr, _region - _addr));
					_size -= (_region - _addr);
					_addr = _region;
				}
				// Get the size of the acc region 
				size_t __size = (_size < i->second->getSize()) ?
						_size : i->second->getSize();
				// If the acc region is affected, invalidate it
				assert(__size > 0);
				acc.push_back(MemRegion((void *)_addr, __size));
				// Update the memory interval we are processing
				_addr += __size;
				_size -= __size;
				// Check for the next region
				i++;
			}
			if(_size) cpu.push_back(MemRegion((void *)_addr, _size));
		}
		MUTEX_UNLOCK(__mutex);
		return ret;
	}

};

};

#endif
