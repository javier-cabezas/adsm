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

#ifndef GMAC_API_CUDA_KERNEL_H_
#define GMAC_API_CUDA_KERNEL_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "config/common.h"
#include "core/Kernel.h"
#include "util/NonCopyable.h"

namespace __impl { namespace cuda {

class Mode;

class KernelConfig;
class KernelLaunch;

class GMAC_LOCAL Argument : public util::ReusableObject<Argument> {
	friend class Kernel;
    const void * ptr_;
    size_t size_;
    long_t offset_;
public:
    Argument(const void * ptr, size_t size, long_t offset);

    const void * ptr() const { return ptr_; }
    size_t size() const { return size_; }
    long_t offset() const { return offset_; }
};



class GMAC_LOCAL Kernel : public core::Kernel {
    friend class KernelLaunch;
protected:
    CUfunction f_;

public:
    Kernel(const core::KernelDescriptor & k, CUmodule mod);
    KernelLaunch * launch(KernelConfig & c);
};

typedef std::vector<Argument> ArgsVector;

class GMAC_LOCAL KernelConfig : public ArgsVector {
protected:
    static const unsigned StackSize_ = 4096;

    uint8_t stack_[StackSize_];
    size_t argsSize_;

    dim3 grid_;
    dim3 block_;
    size_t shared_;

    CUstream stream_;

public:
    /// \todo create a pool of objects to avoid mallocs/frees
    KernelConfig();
    KernelConfig(dim3 grid, dim3 block, size_t shared, cudaStream_t tokens, CUstream stream);
    KernelConfig(const KernelConfig & c);

    void pushArgument(const void * arg, size_t size, long_t offset);

    size_t argsSize() const;
    uint8_t *argsArray();

    dim3 grid() const { return grid_; }
    dim3 block() const { return block_; }
    size_t shared() const { return shared_; }
};

class GMAC_LOCAL KernelLaunch : public core::KernelLaunch,
                                public KernelConfig,
                                public util::NonCopyable {
    friend class Kernel;

protected:
    // \todo Is this really necessary?
    const Kernel & kernel_;
    CUfunction f_;

    KernelLaunch(const Kernel & k, const KernelConfig & c);
public:

    gmacError_t execute();
};

}}

#include "Kernel-impl.h"

#endif
