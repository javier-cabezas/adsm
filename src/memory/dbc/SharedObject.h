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

#ifndef GMAC_MEMORY_TEST_SHAREDOBJECT_H_
#define GMAC_MEMORY_TEST_SHAREDOBJECT_H_

#include "memory/SharedObject.h"

namespace gmac { namespace memory { namespace __dbc {

template<typename T>
class GMAC_LOCAL SharedObject :
    public __impl::SharedObject<T>,
    public virtual gmac::dbc::Contract {
public:
    SharedObject(size_t size, T init);
    virtual ~SharedObject();

    void init();
    void fini();

    // To host functions
    gmacError_t toHost(Block &block) const;
    gmacError_t toHost(Block &block, unsigned blockOff, size_t count) const;
    gmacError_t toHostPointer(Block &block, unsigned blockOff, void *ptr, size_t count) const;
    gmacError_t toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const;

    // To accelerator functions
    gmacError_t toAccelerator(Block &block) const;
    gmacError_t toAccelerator(Block &block, unsigned blockOff, size_t count) const;
    gmacError_t toAcceleratorFromPointer(Block &block, unsigned blockOff, const void *ptr, size_t count) const;
    gmacError_t toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const;

    void *getAcceleratorAddr(void *addr) const;
    Mode &owner() const;
    gmacError_t free();
    gmacError_t realloc(Mode &mode);
};

}}}

#include "SharedObject.ipp"

#endif
