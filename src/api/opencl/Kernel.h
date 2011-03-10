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


#ifndef GMAC_API_OPENCL_KERNEL_H_
#define GMAC_API_OPENCL_KERNEL_H_

#include <CL/cl.h>

#include <list>

#include "config/common.h"
#include "core/Kernel.h"
#include "util/NonCopyable.h"

namespace __impl { namespace opencl {

class Mode;

class KernelConfig;
class KernelLaunch;

class GMAC_LOCAL Argument : public util::ReusableObject<Argument> {
	friend class Kernel;
    const void * ptr_;
    size_t size_;
    unsigned index_;
public:
    Argument(const void * ptr, size_t size, unsigned index);

    const void * ptr() const { return ptr_; }
    size_t size() const { return size_; }
    unsigned index() const { return index_; }
};



class GMAC_LOCAL Kernel : public gmac::core::Kernel { //added
    friend class KernelLaunch;
protected:
    cl_kernel f_;
public:
    Kernel(const core::KernelDescriptor & k, cl_kernel kernel);
    ~Kernel();
    KernelLaunch * launch(KernelConfig & c);
};

typedef std::list<Argument> ArgsList;

class GMAC_LOCAL KernelConfig : protected ArgsList {
protected:
    static const unsigned StackSize_ = 4096;

    uint8_t stack_[StackSize_];
    size_t argsSize_;

    cl_uint workDim_;
    size_t *globalWorkOffset_;
    size_t *globalWorkSize_;
    size_t *localWorkSize_;

    cl_command_queue stream_;

    KernelConfig(const KernelConfig &config);
public:
    /// \todo Remove this piece of shit
    KernelConfig();
    KernelConfig(cl_uint work_dim, size_t *globalWorkOffset, size_t *globalWorkSize, size_t *localWorkSize, cl_command_queue stream);
    ~KernelConfig();

    void setArgument(const void * arg, size_t size, unsigned index);

    KernelConfig &operator=(const KernelConfig &config);

    cl_uint workDim() const { return workDim_; }
    size_t *globalWorkOffset() const { return globalWorkOffset_; }
    size_t *globalWorkSize() const { return globalWorkSize_; }
    size_t *localWorkSize() const { return localWorkSize_; }
};

class GMAC_LOCAL KernelLaunch : public core::KernelLaunch, public KernelConfig, public util::NonCopyable {
    friend class Kernel;

protected:
    cl_kernel f_;

    KernelLaunch(const Kernel & k, const KernelConfig & c);
public:
    ~KernelLaunch();
    gmacError_t execute();
};

}}

#include "Kernel-impl.h"

#endif
