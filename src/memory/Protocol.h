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

#ifndef GMAC_MEMORY_PROTOCOL_H_
#define GMAC_MEMORY_PROTOCOL_H_

#include "config/common.h"
#include "include/gmac/types.h"



namespace __impl {

namespace core {
class IOBuffer;
class Mode;
}
    
namespace memory {

class Block;
class Object;
class Block;

class GMAC_LOCAL Protocol {
public:
    virtual ~Protocol();

    virtual Object *createObject(size_t size, void *cpuPtr, 
        GmacProtection prot, unsigned flags) = 0;
#if 0
    virtual Object *createGlobalObject(size_t size, void *cpuPtr, 
        GmacProtection prot, unsigned flags) = 0;
#endif
    virtual void deleteObject(Object &obj) = 0;

    virtual bool needUpdate(const Block &block) const = 0;

    virtual gmacError_t signalRead(Block &block) = 0;
    virtual gmacError_t signalWrite(Block &block) = 0;

    virtual gmacError_t acquire(Block &block) = 0;
#ifdef USE_VM
    virtual gmacError_t acquireWithBitmap(const Block &block) = 0;
#endif
    virtual gmacError_t release(Block &block) = 0;
    virtual gmacError_t release() = 0;
    virtual gmacError_t remove(Block &block) = 0;

    virtual gmacError_t deleteBlock(Block &block) = 0;

    virtual gmacError_t toHost(Block &block) = 0;
    virtual gmacError_t toDevice(Block &block) = 0;

	virtual gmacError_t copyToBuffer(const Block &block, core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffet, unsigned blockOffset) const = 0;
	
	virtual gmacError_t copyFromBuffer(const Block &block, core::IOBuffer &buffer, size_t size,
		unsigned bufferOffet, unsigned blockOffset) const = 0;

    virtual gmacError_t memset(const Block &block, int v, size_t size, 
        unsigned blockOffset) const = 0;

    virtual gmacError_t copyFromMemory(const Block &block, const void *src, size_t size,
        unsigned blockOffset) const = 0;
    virtual gmacError_t copyFromObject(const Block &block, const Object &object, size_t size,
        unsigned blockOffset) const = 0;
    virtual gmacError_t copyToMemory(const Block &block, void *dst, size_t size,
        unsigned blockOffset) const = 0;

#if 0
    virtual gmacError_t copy(const Object &dstObj, unsigned dstOff, const Object &srcObj, unsigned srcOff, size_t n) = 0;
    virtual gmacError_t memset(const Object &obj, unsigned objectOff, int c, size_t n) = 0;

    virtual gmacError_t moveTo(Object &obj, core::Mode &mode) = 0;

    virtual gmacError_t removeMode(core::Mode &mode) = 0;
#endif

	typedef gmacError_t (Protocol::*CoherenceOp)(Block &);
	typedef gmacError_t (Protocol::*MemoryOp)(const Block &, core::IOBuffer &, size_t, unsigned, unsigned) const;
};

}}

#endif
