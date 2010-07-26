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

#ifndef __API_CUDADRV_ACCELERATOR_H_
#define __API_CUDADRV_ACCELERATOR_H_

#include <kernel/Accelerator.h>
#include "util/Lock.h"

#include <cuda.h>
#include <vector_types.h>

#include <set>

namespace gmac { namespace gpu {
class Context;
class ModuleDescriptor;

class Accelerator : public gmac::Accelerator {
protected:
	CUdevice _device;
	//bool _async;

	std::set<gpu::Context *> queue;
	
#ifndef USE_MULTI_CONTEXT
#ifdef USE_VM
#ifndef USE_HOSTMAP_VM
    Context * _lastContext;
#endif
#endif

    int _major;
    int _minor;

    CUcontext _ctx;
    util::Lock mutex;
#endif

public:
	Accelerator(int n, CUdevice device);
	~Accelerator();
	CUdevice device() const;

	gmac::Context *create();
	void destroy(gmac::Context * ctx);
	size_t nContexts() const;

    gmacError_t bind(gmac::Context * ctx);
#ifdef USE_MULTI_CONTEXT
    CUcontext createCUDAContext();
    void destroyCUDAContext(CUcontext ctx);
#else
    void pushLock();
    void popUnlock();
#endif

#ifdef USE_VM
#ifndef USE_HOSTMAP_VM
    Context *lastContext();
    void setLastContext(Context * ctx);
#endif
#endif
    //bool async() const;

    int major() const;
    int minor() const;
};

}}

#include "Accelerator.ipp"

#endif
