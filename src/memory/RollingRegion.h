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

#include "Region.h"
#include "ProtRegion.h"

#include <config.h>
#include <debug.h>

#include <stdlib.h>

#include <map>
#include <set>

namespace gmac { namespace memory {
class RollingManager;
class ProtSubRegion;
class RollingRegion : public Region {
public:
	typedef std::set<ProtSubRegion *> List;
protected:
	RollingManager &manager;

	// Set of all sub-regions forming the region
	typedef std::map<const void *, ProtSubRegion *> Map;
	Map map;

	// List of sub-regions that are present in memory
	List memory;

	size_t cacheLine;
	size_t offset;

	friend class ProtSubRegion;
	inline void push(ProtSubRegion *region) { memory.insert(region); }

public:
	RollingRegion(RollingManager &manager, void *, size_t, size_t);
	~RollingRegion();

	inline virtual void relate(Context *ctx) {
		assert(ctx->copyToDevice(Manager::ptr(start()), start(), size()) == gmacSuccess);
		_relatives.push_back(ctx);
	}

	ProtSubRegion *find(const void *);
	virtual void invalidate();
	void invalidate(const void *, size_t);
	void flush(const void *, size_t);
};

class ProtSubRegion : public ProtRegion {
protected:
	RollingRegion *parent;
	friend class RollingRegion;
	void silentInvalidate() { _present = _dirty = false; }
public:
	ProtSubRegion(RollingRegion *parent, void *addr, size_t size) :
		ProtRegion(addr, size),
		parent(parent)
	{ }
	~ProtSubRegion() { TRACE("SubRegion %p released", _addr); }

	// Override this methods to insert the regions in the list
	// of sub-regions present in memory
	virtual void readOnly() {
		if(present() == false) parent->push(this);
		ProtRegion::readOnly();
	}

	virtual void readWrite() {
		if(present() == false) parent->push(this);
		ProtRegion::readWrite();
	}
};

} };

#endif
