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

#include "memory/Handler.h"
#include "memory/Object.h"
#include "memory/Protocol.h"
#include "memory/StateObject.h"

#include "util/Logger.h"

namespace gmac {

class IOBuffer;

namespace memory { namespace protocol {

class List;

class Lazy : public Protocol, Handler, protected std::map<Mode *, List *>, util::RWLock {
public:
    typedef enum {
        Invalid,
        ReadOnly,
        Dirty 
    } State;
protected:
    static List GlobalCache_;
    unsigned _maxListSize;

    gmacError_t addDirty(const StateObject<State> &object, SystemBlock<State> &block, bool checkOverflow = true);

    gmacError_t release(const StateObject<State> &object, SystemBlock<State> &block);


    gmacError_t copyHostToDirty(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                                const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count);
    gmacError_t copyHostToReadOnly(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                                   const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count);
    gmacError_t copyHostToInvalid(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                                  const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count);

    gmacError_t copyAcceleratorToDirty(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                                       const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count);
    gmacError_t copyAcceleratorToReadOnly(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                                          const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count);
    gmacError_t copyAcceleratorToInvalid(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                                         const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count);
public:
    Lazy(unsigned limit);
    virtual ~Lazy();

    // Protocol Interface
    Object *createObject(size_t size);
    void deleteObject(const Object &obj);
#ifndef USE_MMAP
    Object *createReplicatedObject(size_t size);
    bool requireUpdate(Block &block);
#endif

    gmacError_t signalRead(const Object &obj, void *addr);
    gmacError_t signalWrite(const Object &obj, void *addr);

    gmacError_t acquire(const Object &obj);
#ifdef USE_VM
    gmacError_t acquireWithBitmap(const Object &obj);
#endif
    virtual gmacError_t release();

    gmacError_t toHost(const Object &obj);
    gmacError_t toDevice(const Object &obj);

    gmacError_t toIOBuffer(IOBuffer &buffer, unsigned bufferOff, const Object &obj, unsigned objectOff, size_t n);
    gmacError_t fromIOBuffer(const Object &obj, unsigned objectOff, IOBuffer &buffer, unsigned bufferOff, size_t n);

    gmacError_t toPointer(void *dst, const Object &objSrc, unsigned objectOff, size_t n);
    gmacError_t fromPointer(const Object &dstObj, unsigned objectOff, const void *src, size_t n);

    gmacError_t copy(const Object &dstObj, unsigned offDst, const Object &objSrc, unsigned offSrc, size_t n);
    gmacError_t memset(const Object &obj, unsigned objectOff, int c, size_t n);

    gmacError_t moveTo(Object &obj, Mode &mode);
};

class Entry {
public:
    const StateObject<Lazy::State> &object;
    SystemBlock<Lazy::State> *block;

    Entry(const StateObject<Lazy::State> &object,
            SystemBlock<Lazy::State> *block) :
        object(object), block(block) {};


    inline void lock() const { block->lock(); }
    inline void unlock() const { block->unlock(); }
};

class List : protected std::list<Entry>, util::RWLock {
    friend class Lazy;
public:
    List() : util::RWLock("List") {};

    void purge(const StateObject<Lazy::State> &object);
    void push(const StateObject<Lazy::State> &object,
            SystemBlock<Lazy::State> *block);
    Entry pop();

    bool empty() const;
    size_t size() const;
};


} } }


#endif
