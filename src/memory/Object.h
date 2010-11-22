/* Copyright (c) 2009, 2010 University of Illinois
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

#ifndef GMAC_MEMORY_OBJECT_H_
#define GMAC_MEMORY_OBJECT_H_

#include <map>

#include "config/common.h"
#include "include/gmac/types.h"

#include "util/Lock.h"
#include "util/Reference.h"
#include "memory/Protocol.h"

namespace __impl { 

namespace core {
	class Mode;
}

namespace memory {

class GMAC_LOCAL Object: protected gmac::util::RWLock, public util::Reference {
protected:
    uint8_t *addr_;
    size_t size_;

	bool valid_;

	typedef std::map<uint8_t *, Block *> BlockMap;
	BlockMap blocks_;

	gmacError_t coherenceOp(Protocol::CoherenceOp op) const;
	gmacError_t memoryOp(Protocol::MemoryOp op, core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset, unsigned objectOffset) const;

	Object(void *addr, size_t size);
	virtual ~Object();
public:

    uint8_t *addr() const;
    uint8_t *end() const;
    size_t size() const;
	bool valid() const;

    virtual void *deviceAddr(const void *addr) const = 0;
	virtual core::Mode &owner(const void *addr) const = 0;
    
	virtual bool addOwner(core::Mode &owner) = 0;
	virtual void removeOwner(const core::Mode &owner) = 0;

	gmacError_t acquire() const;
	gmacError_t toHost() const;
	gmacError_t toDevice() const;

	gmacError_t signalRead(void *addr) const;
	gmacError_t signalWrite(void *addr) const;

	gmacError_t copyToBuffer(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned objectOffset = 0) const;	
	gmacError_t copyFromBuffer(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned objectOffset = 0) const;
};

}}

#include "Object-impl.h"

#endif
