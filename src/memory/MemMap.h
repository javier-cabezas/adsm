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

#include "MemRegion.h"

#include <assert.h>
#include <map>

namespace gmac {

class MemMap {
protected:
	typedef std::map<const void *, MemRegion *> Map;
	Map __map;
	MUTEX(__mutex);
public:
	typedef Map::iterator iterator;
	typedef Map::const_iterator const_iterator;

	MemMap() { MUTEX_INIT(__mutex); }

	virtual ~MemMap() { clean(); }

	inline void lock() { MUTEX_LOCK(__mutex); }
	inline void unlock() { MUTEX_UNLOCK(__mutex); }
	inline Map::iterator begin() { return __map.begin(); }
	inline Map::iterator end() { return __map.end(); }

	inline void insert(MemRegion *i) {
		MUTEX_LOCK(__mutex);
		__map.insert(Map::value_type(i->end(), i));
		MUTEX_UNLOCK(__mutex);
	}

	inline MemRegion *remove(void *addr) {
		Map::iterator i;
		MUTEX_LOCK(__mutex);
		i = __map.upper_bound(addr);
		if(i == __map.end() || i->second->start() != addr)
			FATAL("Bad free for %p", addr);
		MemRegion *ret = i->second;
		__map.erase(i);
		MUTEX_UNLOCK(__mutex);
		return ret;
	}

	inline void clean() {
		MUTEX_LOCK(__mutex);
		Map::iterator i;
		for(i = __map.begin(); i != __map.end(); i++) {
			TRACE("Cleaning MemRegion %p", i->second);
			delete i->second;
		}
		__map.clear();
		MUTEX_UNLOCK(__mutex);
	}

	template<typename T>
	inline T *find(const void *addr) {
		Map::const_iterator i;
		MUTEX_LOCK(__mutex);
		i = __map.upper_bound(addr);
		MUTEX_UNLOCK(__mutex);
		if(i == __map.end() || i->second->start() > addr) return NULL;
		return dynamic_cast<T *>(i->second);
	}
};

};

#endif
