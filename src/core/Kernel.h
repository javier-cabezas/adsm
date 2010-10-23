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

#ifndef GMAC_CORE_KERNEL_H_
#define GMAC_CORE_KERNEL_H_

#include <vector>


#include "config/common.h"
#include "include/gmac-types.h"
#include "memory/ObjectSet.h"
#include "util/Logger.h"
#include "util/ReusableObject.h"

#include "Descriptor.h"

namespace gmac {

class GMAC_LOCAL Argument : public util::ReusableObject<Argument> {
	friend class Kernel;
public:
    void * ptr_;
    size_t size_;
    off_t  offset_;
    Argument(void * ptr, size_t size, off_t offset) :
        ptr_(ptr), size_(size), offset_(offset) {}
};

typedef std::vector<Argument> ArgVector;

/// \todo create a pool of objects to avoid mallocs/frees
class GMAC_LOCAL KernelConfig : public ArgVector, public util::Logger {
protected:
    static const unsigned StackSize_ = 4096;

    char stack_[StackSize_];
    off_t argsSize_;

    KernelConfig(const KernelConfig & c);
public:
    /// \todo create a pool of objects to avoid mallocs/frees
    KernelConfig() : argsSize_(0) {}
    virtual ~KernelConfig() { clear(); }

    void pushArgument(const void * arg, size_t size, off_t offset);
    inline off_t argsSize() const { return argsSize_; }

    inline char * argsArray() { return stack_; }
};

typedef Descriptor<gmacKernel_t> KernelDescriptor;

class KernelLaunch;

class GMAC_LOCAL Kernel : public memory::ObjectSet, public KernelDescriptor
{
public:
    Kernel(const KernelDescriptor & k) :
        KernelDescriptor(k.name(), k.key()) {};
    virtual ~Kernel() {}

    virtual KernelLaunch * launch(KernelConfig & c) = 0;
};

class GMAC_LOCAL KernelLaunch : public memory::ObjectSet {
public:
    virtual ~KernelLaunch() {};
    virtual gmacError_t execute() = 0;
};

}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
