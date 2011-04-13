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


#ifndef GMAC_API_OPENCL_HPE_KERNEL_H_
#define GMAC_API_OPENCL_HPE_KERNEL_H_

#include <CL/cl.h>

#include <list>

#include "config/common.h"
#include "core/hpe/Kernel.h"
#include "util/NonCopyable.h"

namespace __impl { namespace opencl { namespace hpe {

class Mode;

class KernelConfig;
class KernelLaunch;

class GMAC_LOCAL Argument : public util::ReusableObject<Argument> {
	friend class Kernel;
protected:
    size_t size_;

    static const unsigned StackSize_ = 4096;
    uint8_t stack_[StackSize_];
public:
    Argument();
    void setArgument(const void *ptr, size_t size);

    const void * ptr() const { return stack_; }
    size_t size() const { return size_; }
};



class GMAC_LOCAL Kernel : public gmac::core::hpe::Kernel {
    friend class KernelLaunch;
protected:
    cl_kernel f_;
    unsigned nArgs_;
public:
    Kernel(const core::hpe::KernelDescriptor & k, cl_kernel kernel);
    ~Kernel();
    KernelLaunch *launch(Mode &mode, cl_command_queue stream);
};

typedef std::vector<Argument> ArgsVector;

class GMAC_LOCAL KernelConfig : protected ArgsVector {
protected:
    cl_uint workDim_;
    size_t *globalWorkOffset_;
    size_t *globalWorkSize_;
    size_t *localWorkSize_;

public:
    KernelConfig(unsigned nArgs);
    ~KernelConfig();

    void setConfiguration(cl_uint work_dim, size_t *globalWorkOffset,
        size_t *globalWorkSize, size_t *localWorkSize);
    void setArgument(const void * arg, size_t size, unsigned index);

    KernelConfig &operator=(const KernelConfig &config);

    cl_uint workDim() const { return workDim_; }
    size_t *globalWorkOffset() const { return globalWorkOffset_; }
    size_t *globalWorkSize() const { return globalWorkSize_; }
    size_t *localWorkSize() const { return localWorkSize_; }
};

class GMAC_LOCAL KernelLaunch : public core::hpe::KernelLaunch, public KernelConfig, public util::NonCopyable {
    friend class Kernel;

protected:
    cl_kernel f_;
    cl_command_queue stream_;
    cl_event event_;

    KernelLaunch(Mode &mode, const Kernel & k, cl_command_queue stream);
public:
    ~KernelLaunch();
    gmacError_t execute();
    cl_event getCLEvent();
};

}}}

#include "Kernel-impl.h"

#endif
