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

#ifndef GMAC_MEMORY_SHAREDOBJECT_H_
#define GMAC_MEMORY_SHAREDOBJECT_H_

#include "config/common.h"
#include "core/IOBuffer.h"
#include "core/Mode.h"
#include "memory/Block.h"
#include "memory/StateObject.h"

namespace gmac { namespace memory { namespace __impl {

template<typename T>
class GMAC_LOCAL SharedObject : public StateObject<T> {
protected:
    Mode *owner_;
    bool mapped_;
    AcceleratorBlock *accBlock_;
public:
    SharedObject(size_t size, void *cpuPtr, T init);
    virtual ~SharedObject();

    TESTABLE void init();
    TESTABLE void fini();

    // To host functions
    TESTABLE gmacError_t toHost(Block &block) const;
    TESTABLE gmacError_t toHost(Block &block, unsigned blockOff, size_t count) const;
    TESTABLE gmacError_t toHostPointer(Block &block, unsigned blockOff, void *ptr, size_t count) const;
    TESTABLE gmacError_t toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const;

    // To accelerator functions
    TESTABLE gmacError_t toAccelerator(Block &block) const;
    TESTABLE gmacError_t toAccelerator(Block &block, unsigned blockOff, size_t count) const;
    TESTABLE gmacError_t toAcceleratorFromPointer(Block &block, unsigned blockOff, const void *ptr, size_t count) const;
    TESTABLE gmacError_t toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const;

    TESTABLE void *getAcceleratorAddr(void *addr) const;
    TESTABLE Mode &owner() const;

    TESTABLE gmacError_t free();
    TESTABLE gmacError_t realloc(Mode &mode);

    bool isLocal() const;
    bool isInAccelerator() const;
};

}}}

#include "SharedObject.ipp"

#ifdef USE_DBC
#include "memory/dbc/SharedObject.h"
#endif


#endif
