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

#ifndef __API_CUDA_ACCELERATOR_H_
#define __API_CUDA_ACCELERATOR_H_

#include "kernel/Mode.h"
#include "kernel/Accelerator.h"
#include "util/Lock.h"

#include "Module.h"

#include <vector_types.h>

#include <set>

namespace gmac { namespace cuda {
class Mode;
class ModuleDescriptor;

typedef CUstream Stream;

class AcceleratorLock : public util::Lock {
protected:
    friend class Accelerator;
public:
    AcceleratorLock() : Lock("Accelerator") {};
};

class AlignmentMap : public std::map<CUdeviceptr, CUdeviceptr> { };

class Accelerator : public gmac::Accelerator {
protected:
	CUdevice _device;

	std::set<Mode *> _queue;
    AlignmentMap _alignMap;

    int _major;
    int _minor;

#ifndef USE_MULTI_CONTEXT
    CUcontext _ctx;
    AcceleratorLock _mutex;
    ModuleVector _modules;
#endif

    static gmacError_t _error;
public:
	Accelerator(int n, CUdevice device);
	~Accelerator();
	CUdevice device() const;

	gmac::Mode *createMode();
    void destroyMode(gmac::Mode *mode);

#ifdef USE_MULTI_CONTEXT
    CUcontext createCUDAContext();
    void destroyCUDAContext(CUcontext ctx);
#else
    ModuleVector &createModules();
    void switchIn();
    void switchOut();
#endif

    int major() const;
    int minor() const;

	gmacError_t malloc(void **addr, size_t size, unsigned align = 1);
	gmacError_t free(void *addr);

    /* Synchronous interface */
	gmacError_t copyToDevice(void *dev, const void *host, size_t size);
	gmacError_t copyToHost(void *host, const void *dev, size_t size);
	gmacError_t copyDevice(void *dst, const void *src, size_t size);

    /* Asynchronous interface */
    gmacError_t copyToDeviceAsync(void *dec, const void *host, size_t size, Stream stream);
    gmacError_t copyToHostAsync(void *host, const void *dev, size_t size, Stream stream);
    gmacError_t syncStream(Stream stream);

    gmacError_t memset(void *addr, int c, size_t size);

	gmacError_t sync();
	gmac::KernelLaunch * launch(gmacKernel_t kernel);

    gmacError_t hostAlloc(void **addr, size_t size);
    gmacError_t hostFree(void *addr);
    void *hostMap(void *addr);

    static gmacError_t error(CUresult r);
    static CUdeviceptr gpuAddr(void *addr);
    static CUdeviceptr gpuAddr(const void *addr);
};

}}

#include "Accelerator.ipp"

#endif
