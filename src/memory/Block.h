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

#include "core/Mode.h"
#include "include/gmac-types.h"
#include "util/Lock.h"
#include "util/Logger.h"

namespace gmac { namespace memory {

class GMAC_LOCAL Block : public util::Lock, public util::Logger {
protected:
    void *addr_;
    size_t size_;

    Block(void *addr, size_t size);
    inline void *mirrorAddress(void *src) const;
public:
    virtual ~Block() {};

    inline uint8_t *addr() const { return (uint8_t *) addr_; }
    inline uint8_t *end() const { return addr() + size_; }
    inline size_t size() const { return size_; }

    inline void lock() const { return util::Lock::lock(); }
    inline void unlock() const { return util::Lock::unlock(); }
};

class GMAC_LOCAL AcceleratorBlock : public Block {
protected:
    Mode &owner_;
public:
    AcceleratorBlock(Mode &owner, void *addr, size_t size);
    ~AcceleratorBlock();

#if 0
    gmacError_t toDevice(off_t off, Block &block);
    gmacError_t toHost(off_t off, Block &block);
    gmacError_t toHost(off_t off, void *hostAddr, size_t count);
#endif

    Mode &owner() { return owner_; }
};

template<typename T>
class GMAC_LOCAL SystemBlock : public Block {
protected:
    T state_;

public:
    SystemBlock(void *addr, size_t size, T state);
    ~SystemBlock();

    gmacError_t update(off_t off, Block *block);

    T state() const;
    void state(T s);
};

}}

#include "Block.ipp"

#endif
