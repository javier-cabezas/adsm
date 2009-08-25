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
#include <paraver.h>

#include "MemRegion.h"

#include <assert.h>
#include <map>

namespace gmac {

class MemMap {
protected:
	typedef std::map<const void *, MemRegion *> Map;
	Map __map;
	MUTEX(local);

	static Map __global;
	static MUTEX(global);

	static void globalLock() {
		enterLock(mmGlobal);
		MUTEX_LOCK(global);
		exitLock();
	}
	static void globalUnlock() { MUTEX_UNLOCK(global); }

	MemRegion *localFind(const void *addr) {
		Map::const_iterator i;
		MemRegion *ret = NULL;
		i = __map.upper_bound(addr);
		if(i != __map.end() && i->second->start() <= addr) {
			ret = i->second;
		}
		return ret;
	}

	MemRegion *globalFind(const void *addr) {
		Map::const_iterator i;
		MemRegion *ret = NULL;
		i = __global.upper_bound(addr);
		if(i != __global.end() && i->second->start() <= addr)
			ret = i->second;
		return ret;
	}

public:
	typedef Map::iterator iterator;
	typedef Map::const_iterator const_iterator;

	MemMap() { MUTEX_INIT(local); }

	virtual ~MemMap() { clean(); }

	static void init() { MUTEX_INIT(global); }
	inline void lock() { 
		enterLock(mmLocal);
		MUTEX_LOCK(local);
		exitLock();
	}
	inline void unlock() {
		MUTEX_UNLOCK(local);
	}
	inline iterator begin() { return __map.begin(); }
	inline iterator end() { return __map.end(); }


	inline void insert(MemRegion *i) {
		globalLock();
		__map.insert(Map::value_type(i->end(), i));
		__global.insert(Map::value_type(i->end(), i));
		globalUnlock();
	}

	inline MemRegion *remove(void *addr) {
		Map::iterator i;
		globalLock();
		i = __map.upper_bound(addr);
		assert(i != __map.end() && i->second->start() == addr);
		MemRegion *ret = i->second;
		__map.erase(i);
		i = __global.upper_bound(addr);
		assert(i != __global.end() && i->second->start() == addr);
		__global.erase(i);
		globalUnlock();
		return ret;
	}

	inline void clean() {
		globalLock();
		Map::iterator i;
		for(i = __map.begin(); i != __map.end(); i++) {
			TRACE("Cleaning MemRegion %p", i->second);
			__global.erase(i->first);
			delete i->second;
		}
		__map.clear();
		globalUnlock();
	}

	template<typename T>
	inline T *find(const void *addr) {
		MemRegion *ret = NULL;
		lock();
		ret = localFind(addr);
		if(ret == NULL) {
			globalLock();
			ret = globalFind(addr);
			globalUnlock();
		}
		unlock();
		return dynamic_cast<T *>(ret);
	}
};

};

#endif
