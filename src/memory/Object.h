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

#ifndef __MEMORY_OBJECT_H_
#define __MEMORY_OBJECT_H_

#include <config.h>
#include <gmac/gmac.h>

#include <memory/Block.h>
#include <memory/Protocol.h>

#include <kernel/Mode.h>

#include <util/Lock.h>
#include <util/Logger.h>

#include <set>


namespace gmac { namespace memory {

class Object: protected util::RWLock, public util::Logger {
private:
#ifdef USE_MMAP
#ifdef ARCH_32BIT
    static const size_t mmSize = 0;
#else
	static const size_t mmSize = 0x10000000000;
#endif
#endif

protected:
    void *__addr;
    size_t __size;

    Object(void *__addr, size_t __size);

    static void *map(void *addr, size_t size);
    static void unmap(void *addr, size_t size);
public:
    virtual ~Object() {};

    inline void *addr() const { return __addr; };
    inline size_t size() const { return __size; };

    inline void *start() const { return __addr; }
    inline void *end() const {
        return (void *)((uint8_t *)__addr + __size);
    }

    virtual gmacError_t acquire(Block *block) = 0;
    virtual gmacError_t release(Block *block) = 0;

    virtual Mode *owner() const = 0;
    virtual void *device(void *addr) = 0;
};

typedef std::set<Object *> ObjectSet;

class DistributedObject {
public:
    virtual ~DistributedObject() {};
    virtual gmacError_t addOwner(Mode *mode) = 0;
    virtual gmacError_t removeOwner(Mode *mode) = 0;
};

template<typename T>
class StateObject: public Object {
public:
    typedef std::map<void *, SystemBlock<T> *> SystemMap;
protected:
    SystemMap systemMap;
    void setupSystem(T init);
public:
    StateObject(size_t size);
    virtual ~StateObject();

    SystemBlock<T> *findBlock(void *addr);
    inline SystemMap &blocks() { return systemMap; }

    virtual void state(T s);
};


template<typename T>
class SharedObject : public StateObject<T> {
protected:
    Mode *__owner;

    AcceleratorBlock *accelerator;
public:
    SharedObject(size_t size, T init);
    virtual ~SharedObject();

    virtual gmacError_t acquire(Block *block);
    virtual gmacError_t release(Block *block);

    virtual void *device(void *addr);
    inline virtual Mode *owner() const { return __owner; }
};


#ifndef USE_MMAP
template<typename T>
class ReplicatedObject : public StateObject<T>, public DistributedObject {
protected:
    class AcceleratorMap: public std::map<Mode *, AcceleratorBlock *>, util::RWLock {
    protected:
        friend class ReplicatedObject<T>;
    public:
        AcceleratorMap() : util::RWLock(paraver::LockObject) {};
    };

    AcceleratorMap accelerator;
public:
    ReplicatedObject(size_t size, T init);
    virtual ~ReplicatedObject();

    virtual gmacError_t acquire(Block *block);
    virtual gmacError_t release(Block *block);

    virtual void *device(void *addr);
    inline virtual Mode *owner() const { return gmac::Mode::current(); }

    gmacError_t addOwner(Mode *mode);
    gmacError_t removeOwner(Mode *mode);
};

class CentralizedObject : public Object {
protected:
public:
    CentralizedObject(size_t __size);
    virtual ~CentralizedObject();

    virtual void *device(void *addr);
    inline virtual Mode *owner() const { return gmac::Mode::current(); }

    inline gmacError_t acquire(Block *block) {
        fatal("Trying to acquire a centralized object");
        return gmacErrorInvalidValue;
    }
    inline gmacError_t release(Block *block) {
        fatal("Trying to release a centralized object");
        return gmacErrorInvalidValue;
    }

};
#endif

} }

#include "Object.ipp"

#endif
