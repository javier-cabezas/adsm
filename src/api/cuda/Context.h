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

#ifndef GMAC_API_CUDA_CONTEXT_H_
#define GMAC_API_CUDA_CONTEXT_H_

#include <cuda.h>
#include <vector_types.h>

#include <map>
#include <vector>

#include "config/common.h"
#include "config/config.h"

#include "core/Context.h"
#include "util/Lock.h"

#include "Kernel.h"

namespace __impl {

namespace core {
    class IOBuffer;
}

namespace cuda {

class Accelerator;

class GMAC_LOCAL Context : public __impl::core::Context {
protected:
    static void * FatBin_;
	static const unsigned USleepLaunch_ = 100;

	typedef std::map<void *, void *> AddressMap;
	static AddressMap HostMem_;

    CUstream streamLaunch_;
    CUstream streamToAccelerator_;
    CUstream streamToHost_;
    CUstream streamAccelerator_;

    Mode &mode_;
    core::IOBuffer *buffer_;

    KernelConfig call_;

    void setupCUstreams();
    void cleanCUstreams();
    gmacError_t syncCUstream(CUstream);

public:
	Context(Accelerator &acc, Mode &mode);
	~Context();

	gmacError_t copyToAccelerator(void *dev, const void *host, size_t size);
	gmacError_t copyToHost(void *host, const void *dev, size_t size);
	gmacError_t copyAccelerator(void *dst, const void *src, size_t size);

    gmacError_t memset(void *addr, int c, size_t size);

    core::KernelLaunch &launch(core::Kernel &kernel);
    gmacError_t sync();

    gmacError_t bufferToAccelerator(void *dst, core::IOBuffer &buffer, size_t size, off_t off = 0);
    gmacError_t acceleratorToBuffer(core::IOBuffer &buffer, const void *dst, size_t size, off_t off = 0);
    gmacError_t waitAccelerator();

    void call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens);
	void argument(const void *arg, size_t size, off_t offset);

    const CUstream eventStream() const;

    Accelerator & accelerator();
    gmacError_t waitForBuffer(core::IOBuffer &buffer);
};

}}

#include "Context-impl.h"

#endif
