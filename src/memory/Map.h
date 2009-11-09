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

#include <memory/PageTable.h>
#include <memory/Region.h>

#include <assert.h>
#include <map>

namespace gmac { namespace memory {

class Map {
protected:
	typedef std::map<const void *, Region *> __Map;
	__Map __map;
	MUTEX(local);

	static __Map *__global;
	static unsigned count;
	static MUTEX(global);

	static void globalLock() {
		enterLock(mmGlobal);
		MUTEX_LOCK(global);
		exitLock();
	}
	static void globalUnlock() { MUTEX_UNLOCK(global); }

	Region *localFind(const void *addr) {
		__Map::const_iterator i;
		Region *ret = NULL;
		i = __map.upper_bound(addr);
		if(i != __map.end() && i->second->start() <= addr) {
			ret = i->second;
		}
		return ret;
	}

	Region *globalFind(const void *addr) {
		__Map::const_iterator i;
		Region *ret = NULL;
		i = __global->upper_bound(addr);
		if(i != __global->end() && i->second->start() <= addr)
			ret = i->second;
		return ret;
	}

	inline void clean() {
		__Map::iterator i;
		for(i = __map.begin(); i != __map.end(); i++) {
			TRACE("Cleaning Region %p", i->second);
			__global->erase(i->first);
			delete i->second;
		}
		__map.clear();
	}


	PageTable __pageTable;

public:
	typedef __Map::iterator iterator;
	typedef __Map::const_iterator const_iterator;

	Map() {
		MUTEX_INIT(local);
		globalLock();
		if(__global == NULL) __global = new __Map();
		count++;
		globalUnlock();
	}

	virtual ~Map() {
		TRACE("Cleaning Memory Map");
		globalLock();
		clean();
		count--;
		if(count == 0) delete __global;
		globalUnlock();
	}

	static void init() { MUTEX_INIT(global); }

	inline void realloc() { __pageTable.realloc(); }

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


	inline void insert(Region *i) {
		globalLock();
		__map.insert(__Map::value_type(i->end(), i));
		__global->insert(__Map::value_type(i->end(), i));
		globalUnlock();
	}

	Region *remove(void *addr);

	inline PageTable &pageTable() { return __pageTable; }
	inline const PageTable &pageTable() const { return __pageTable; }

	template<typename T>
	inline T *find(const void *addr) {
		Region *ret = NULL;
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

}}

#endif
