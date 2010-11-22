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

#ifndef GMAC_MEMORY_BLOCK_H_
#define GMAC_MEMORY_BLOCK_H_

#include "config/common.h"
#include "config/config.h"

#include "include/gmac/types.h"
#include "memory/Protocol.h"
#include "util/Reference.h"

namespace __impl { 

namespace core {
	class Mode;
	class IOBuffer;
}

namespace memory {

class GMAC_LOCAL Block : public gmac::util::Lock, public util::Reference {
protected:
	Protocol &protocol_;

	size_t size_;
	uint8_t *addr_;
	uint8_t *shadow_;

	Block(Protocol &protocol, uint8_t *addr, uint8_t *shadow, size_t size);
    virtual ~Block();
public:
    uint8_t *addr() const;
	size_t size() const;

	gmacError_t signalRead();
	gmacError_t signalWrite();
	gmacError_t coherenceOp(Protocol::CoherenceOp op);
	gmacError_t memoryOp(Protocol::MemoryOp op, core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset, unsigned blockOffset);
    gmacError_t memset(int v, size_t size, unsigned blockOffset = 0) const;

	virtual core::Mode &owner() const = 0;
	virtual void *deviceAddr(const void *addr) const = 0;

	virtual gmacError_t toHost() const = 0;
	virtual gmacError_t toDevice() const = 0;

	virtual gmacError_t copyToHost(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const = 0;
	virtual gmacError_t copyToDevice(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const = 0;
	
	virtual gmacError_t copyFromHost(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const = 0;
	virtual gmacError_t copyFromDevice(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const = 0;
    
    virtual gmacError_t hostMemset(int v, size_t size,
        unsigned blockOffset = 0) const = 0;
    virtual gmacError_t deviceMemset(int v, size_t size, 
        unsigned blockOffset = 0) const = 0;
    
    
};


}}

#include "Block-impl.h"

#ifdef USE_DBC
//#include "memory/dbc/Block.h"
#endif

#endif
