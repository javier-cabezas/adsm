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

#ifndef GMAC_API_CUDA_MODE_H_
#define GMAC_API_CUDA_MODE_H_

#include <config.h>

#include "Context.h"

#include "core/Mode.h"
#include "core/IOBuffer.h"

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>

namespace gmac { namespace cuda {

class ContextLock : public util::Lock {
    friend class Mode;
public:
    ContextLock() : util::Lock("Context") {};
};

class Texture;
class Accelerator;

class Mode : public gmac::Mode {
    friend class Switch;
protected:
#ifdef USE_MULTI_CONTEXT
    CUcontext _cudaCtx;
#endif
    void switchIn();
    void switchOut();

    gmac::Context &getContext();

#ifdef USE_VM
    CUdeviceptr bitmapDevPtr_;
    CUdeviceptr bitmapShiftPageDevPtr_;
#ifdef BITMAP_BIT
    CUdeviceptr bitmapShiftEntryDevPtr_;
#endif
#endif

#ifdef USE_MULTI_CONTEXT
	ModuleVector modules;
#else
    ModuleVector *modules;
#endif

public:
    Mode(Process &proc, Accelerator &acc);
    ~Mode();

    gmacError_t hostAlloc(void **addr, size_t size);
    gmacError_t hostFree(void *addr);
    void *hostMap(void *addr);

	gmacError_t execute(gmac::KernelLaunch &launch);

    gmacError_t bufferToDevice(void *dst, gmac::IOBuffer &buffer, size_t size, off_t off = 0);
    gmacError_t deviceToBuffer(gmac::IOBuffer &buffer, const void *src, size_t size, off_t off = 0);

    void call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens);
	void argument(const void *arg, size_t size, off_t offset);

    const Variable *constant(gmacVariable_t key) const;
    const Variable *variable(gmacVariable_t key) const;
    const Texture *texture(gmacTexture_t key) const;

    CUstream eventStream();

    static Mode & current();
    Accelerator &accelerator();

#ifdef USE_VM
    CUdeviceptr dirtyBitmapDevPtr() const;
    CUdeviceptr dirtyBitmapShiftPageDevPtr() const;
#ifdef BITMAP_BIT
    CUdeviceptr dirtyBitmapShiftEntryDevPtr() const;
#endif
#endif

    gmacError_t waitForBuffer(IOBuffer &buffer);
};

}}

#include "Mode.ipp"

#endif
