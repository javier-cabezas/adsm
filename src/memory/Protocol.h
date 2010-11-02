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
#include "include/gmac-types.h"
#include "util/Logger.h"

namespace gmac {
class IOBuffer;
class Mode;
}

namespace gmac { namespace memory {

class Block;
class Object;

class GMAC_LOCAL Protocol : public util::Logger {
public:
    virtual ~Protocol() {};

    virtual Object *createSharedObject(size_t size, void *cpuPtr, GmacProtection prot) = 0;
    virtual void deleteObject(const Object &obj) = 0;
#ifndef USE_MMAP
    virtual Object *createReplicatedObject(size_t size) = 0;
    virtual Object *createCentralizedObject(size_t size)
;
    virtual bool requireUpdate(Block &block) = 0;
#endif

    virtual gmacError_t signalRead(const Object &obj, void *addr) = 0;
    virtual gmacError_t signalWrite(const Object &obj, void *addr) = 0;

    virtual gmacError_t acquire(const Object &obj) = 0;
#ifdef USE_VM
    virtual gmacError_t acquireWithBitmap(const Object &obj) = 0;
#endif
    virtual gmacError_t release() = 0;

    virtual gmacError_t toHost(const Object &obj) = 0;
    virtual gmacError_t toDevice(const Object &obj) = 0;

    virtual gmacError_t toIOBuffer(IOBuffer &buffer, unsigned bufferOff, const Object &obj, unsigned objectOff, size_t n) = 0;
    virtual gmacError_t fromIOBuffer(const Object &obj, unsigned objectOff, IOBuffer &buffer, unsigned bufferOff, size_t n) = 0;

    virtual gmacError_t toPointer(void *dst, const Object &srcObj, unsigned objectOff, size_t n) = 0;
    virtual gmacError_t fromPointer(const Object &dstObj, unsigned objectOff, const void *src, size_t n) = 0;

    virtual gmacError_t copy(const Object &dstObj, unsigned dstOff, const Object &srcObj, unsigned srcOff, size_t n) = 0;
    virtual gmacError_t memset(const Object &obj, unsigned objectOff, int c, size_t n) = 0;

    virtual gmacError_t moveTo(Object &obj, Mode &mode) = 0;

    virtual gmacError_t removeMode(Mode &mode) = 0;

    //virtual gmacError_t toHost(Object &obj) = 0;
};

}}

#endif
