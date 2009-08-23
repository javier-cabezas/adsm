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

#ifndef __MEMORY_MEMREGION_H_
#define __MEMORY_MEMREGION_H_

#include <config.h>
#include <threads.h>
#include <debug.h>

#include <memory/os/Memory.h>

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <signal.h>

#include <assert.h>

#include <iostream>
#include <list>

namespace gmac {

typedef unsigned long addr_t;

//! Generic Memory Region Descriptor
class Context;
class MemRegion {
private:
	Context *_context;
protected:
	//! Starting memory address for the region
	addr_t _addr;
	//! Size in bytes of the region
	size_t _size;

	inline addr_t __addr(void *addr) const { return (addr_t)addr; }
	inline addr_t __addr(const void *addr) const { return (addr_t)addr; }
	inline void * __void(addr_t addr) const { return (void *)addr; }

public:
	//! Constructor
	//! \param addr Start memory address
	//! \param size Size in bytes
	MemRegion(void *addr, size_t size);

	virtual ~MemRegion() {};

	inline Context *context() { return _context; }

	//! Returns the size (in bytes) of the Region
	inline size_t size() const { return _size; }
	//! Sets the size (in bytes) of the Region
	inline void size(size_t size) { _size = size; }
	//! Returns the address of the Region
	inline void *start() const { return __void(_addr); }
	//! Returns the end address of the region
	inline void *end() const { return __void(_addr + _size); }
	//! Sets the address of the Region
	inline void start(void *addr) { _addr = __addr(addr); }
};

typedef std::list<MemRegion> RegionList;

};

#endif
