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

#ifndef __MEMORY_PROTOCOL_LAZY_H_
#define __MEMORY_PROTOCOL_LAZY_H_

#include <gmac/gmac.h>

#include <memory/Protocol.h>
#include <memory/Handler.h>
#include <memory/Object.h>

#include <util/Logger.h>

namespace gmac { namespace memory { namespace protocol {

class Lazy : public gmac::memory::Protocol, public gmac::memory::Handler {
public:
    typedef enum {
        Invalid,
        ReadOnly,
        Dirty 
    } State;
protected:

public:
    Lazy() {};
    ~Lazy() {};

    // Protocol Interface
    Object *createObject(size_t size);
#ifndef USE_MMAP
    Object *createReplicatedObject(size_t size);
    bool requireUpdate(Block *block);
#endif

    gmacError_t read(Object &obj, void *addr);
    gmacError_t write(Object &obj, void *addr);

    gmacError_t acquire(Object &obj);
#ifdef USE_VM
    gmacError_t acquireWithBitmap(Object &obj);
#endif
    gmacError_t release(Object &obj);

    gmacError_t toHost(Object &obj);
    gmacError_t toDevice(Object &obj);

    gmacError_t toIOBuffer(IOBuffer *buffer, Object &obj, void *addr, size_t n);
    gmacError_t fromIOBuffer(IOBuffer *buffer, Object &obj, void *addr, size_t n);

    gmacError_t toPointer(void *dst, const void *src, const Object &srcObj, size_t n);
    gmacError_t fromPointer(void *dst, const void *src, Object &dstObj, size_t n);

    gmacError_t copy(void *dst, const void *src, Object &dstObj, const void *srcObj, size_t n);
    gmacError_t memset(void *s, int c, size_t n);
};


} } }


#endif
