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

#ifndef GMAC_MEMORY_STATEOBJECT_H_
#define GMAC_MEMORY_STATEOBJECT_H_

#include <map>

#include "config/common.h"

#include "memory/Object.h"
#include "memory/SystemBlock.h"

namespace __impl { namespace memory {

template<typename T>
class GMAC_LOCAL StateObject: public __impl::memory::Object {
public:
    typedef std::map<void *, SystemBlock<T> *> SystemMap;
protected:
    T init_;
    SystemMap systemMap;
    void setupSystem();
public:
    StateObject(size_t size, T init);
    virtual ~StateObject();

    // To host functions
    virtual gmacError_t toHost(memory::Block &block) const = 0;
    virtual gmacError_t toHost(memory::Block &block, unsigned blockOff, size_t count) const = 0;
    virtual gmacError_t toHostPointer(memory::Block &block, unsigned blockOff, void *ptr, size_t count) const = 0;
    virtual gmacError_t toHostBuffer(memory::Block &block, unsigned blockOff, core::IOBuffer &buffer, unsigned bufferOff, size_t count) const = 0;

    // To accelerator functions
    virtual gmacError_t toAccelerator(memory::Block &block) const = 0;
    virtual gmacError_t toAccelerator(memory::Block &block, unsigned blockOff, size_t count) const = 0;
    virtual gmacError_t toAcceleratorFromPointer(memory::Block &block, unsigned blockOff, const void *ptr, size_t count) const = 0;
    virtual gmacError_t toAcceleratorFromBuffer(memory::Block &block, unsigned blockOff, core::IOBuffer &buffer, unsigned bufferOff, size_t count) const = 0;

    SystemBlock<T> *findBlock(const void *addr) const;
    typename SystemMap::iterator getBlockIterator(const void *addr);
    typename SystemMap::const_iterator getBlockIterator(const void *addr) const;
    inline SystemMap &blocks();
    inline const SystemMap &blocks() const;

    virtual void state(T s);
    virtual gmacError_t memsetAccelerator(memory::Block &block, unsigned blockOff, int c, size_t count) const = 0;
};

} }

#include "StateObject.ipp"

#endif
