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

#ifndef __MEMORY_CACHEREGION_H_
#define __MEMORY_CACHEREGION_H_

#include "MemRegion.h"
#include "ProtRegion.h"

#include <config/config.h>
#include <config/debug.h>

#include <stdlib.h>
#include <list>

namespace gmac {
class CacheManager;
class ProtSubRegion;
class CacheRegion : public ProtRegion {
protected:
	CacheManager &manager;

	// Set of all sub-regions forming the region
	typedef HASH_MAP<void *, ProtSubRegion *> Set;
	Set set;

	// List of sub-regions that are present in memory
	typedef std::list<ProtSubRegion *> List;
	List memory;

	size_t cacheLine;
	size_t offset;

	template<typename T>
	inline T powerOfTwo(T k) {
		if(k == 0) return 1;
		for(int i = 1; i < sizeof(T) * 8; i <<= 1)
			k = k | k >> i;
		return k + 1;
	}

	friend class ProtSubRegion;
	inline void present(ProtSubRegion *region) { memory.push_back(region); }

public:
	CacheRegion(CacheManager &manager, void *, size_t, size_t);
	~CacheRegion();

	inline ProtSubRegion *find(const void *addr) {
		void *base = (void *)(((unsigned long)addr & ~(cacheLine - 1)) + offset);
		Set::const_iterator i = set.find(base);
		if(i == set.end()) return NULL;
		return i->second;
	}

	void invalidate();
};

class ProtSubRegion : public ProtRegion {
protected:
	CacheRegion *parent;
	friend class CacheRegion;
	void silentInvalidate() { present = dirty = false; }
public:
	ProtSubRegion(CacheRegion *parent, void *addr, size_t size) :
		ProtRegion(addr, size),
		parent(parent)
	{ }
	~ProtSubRegion() { TRACE("SubRegion %p released", addr); }

	bool belongs(CacheRegion *r) const { parent == r; }

	virtual void invalidate() {
		FATAL("Invalidation on a SubRegion %p", addr);
	}

	// Override this methods to insert the regions in the list
	// of sub-regions present in memory
	virtual void readOnly() {
		parent->present(this);
		ProtRegion::readOnly();
	}

	virtual void readWrite() {
		parent->present(this);
		ProtRegion::readWrite();
	}
};

};

#endif
