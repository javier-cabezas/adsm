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

#ifndef __MEMORY_BLOCK_H_
#define __MEMORY_BLOCK_H_

#include "config.h"

#include "gmac/gmac.h"

#include "util/Lock.h"
#include "util/Logger.h"

#include "core/Mode.h"

namespace gmac {

namespace memory {


class Block: public util::Lock, public util::Logger {
protected:
    void *__addr;
    size_t __size;

    Block(void *addr, size_t size);
    inline void *mirrorAddress(void *src) const;
public:
    virtual ~Block() {};

    inline void *addr() const { return __addr; }
    inline size_t size() const { return __size; }

    inline void lock() const { return util::Lock::lock(); }
    inline void unlock() const { return util::Lock::unlock(); }
};

class AcceleratorBlock : public Block {
protected:
    Mode &_owner;
public:
    AcceleratorBlock(Mode &owner, void *addr, size_t size);
    ~AcceleratorBlock();

    gmacError_t toDevice(off_t off, Block &block);
    gmacError_t toHost(off_t off, Block &block);
    gmacError_t toHost(off_t off, void *hostAddr, size_t count);
};

template<typename T>
class SystemBlock : public Block {
protected:
    T _state;

public:
    SystemBlock(void *addr, size_t size, T state);
    ~SystemBlock();

    gmacError_t update(off_t off, Block *block);

    T state() const;
    void state(T s);

    inline void *addr() const { return __addr; }
};

}}

#include "Block.ipp"

#endif
