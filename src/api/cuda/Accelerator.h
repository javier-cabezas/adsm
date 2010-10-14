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
WITH THE SOFTWARE.
*/

#ifndef GMAC_API_CUDA_ACCELERATOR_H_
#define GMAC_API_CUDA_ACCELERATOR_H_

#include "core/Mode.h"
#include "core/Accelerator.h"
#include "util/Lock.h"
#include "trace/Function.h"

#include "Module.h"

#include <vector_types.h>

#include <set>

namespace gmac { namespace cuda {
class Mode;
class ModuleDescriptor;

class Switch {
public:
    static void in();
    static void out();
};

typedef CUstream Stream;

class AcceleratorLock : public util::Lock {
    friend class Accelerator;
public:
    AcceleratorLock() : Lock("Accelerator") {}
};

class AlignmentMap : public std::map<CUdeviceptr, CUdeviceptr>, public util::RWLock { 
    friend class Accelerator;
public:
    AlignmentMap() : RWLock("Aligment") {}
    ~AlignmentMap() { lockWrite(); }
};

class Accelerator : public gmac::Accelerator {
    friend class Switch;
protected:
	CUdevice device_;

	std::set<Mode *> _queue;
    AlignmentMap _alignMap;

    int _major;
    int _minor;

#ifdef USE_MULTI_CONTEXT
    static gmac::util::Private<CUcontext> _Ctx;
#else
    CUcontext _ctx;
    AcceleratorLock _mutex;
    ModuleVector _modules;
#endif

    void pushContext() const;
    void popContext() const;

public:
	Accelerator(int n, CUdevice device);
	~Accelerator();
	CUdevice device() const;

    static void init();

	gmac::Mode *createMode(gmac::Process &proc);
    void registerMode(gmac::Mode &mode);
    void unregisterMode(gmac::Mode &mode);

#ifdef USE_MULTI_CONTEXT
    CUcontext createCUcontext();
    void destroyCUcontext(CUcontext ctx);

    void setCUcontext(CUcontext *ctx);
    ModuleVector createModules();
    void destroyModules(ModuleVector & modules);
#else
    ModuleVector *createModules();
#endif

    int major() const;
    int minor() const;

	gmacError_t malloc(void **addr, size_t size, unsigned align = 1);
	gmacError_t free(void *addr);

    /* Synchronous interface */
	gmacError_t copyToAccelerator(void *dev, const void *host, size_t size);
	gmacError_t copyToHost(void *host, const void *dev, size_t size);
	gmacError_t copyAccelerator(void *dst, const void *src, size_t size);

    /* Asynchronous interface */
    gmacError_t copyToAcceleratorAsync(void *dec, const void *host, size_t size, CUstream stream);
    gmacError_t copyToHostAsync(void *host, const void *dev, size_t size, CUstream stream);
    CUstream createCUstream();
    void destroyCUstream(CUstream stream);
    CUresult queryCUstream(CUstream stream);
    gmacError_t syncCUstream(CUstream stream);

    gmacError_t execute(KernelLaunch &launch);

    gmacError_t memset(void *addr, int c, size_t size);

	gmacError_t sync();

    gmacError_t hostAlloc(void **addr, size_t size);
    gmacError_t hostFree(void *addr);
    void *hostMap(void *addr);

    static gmacError_t error(CUresult r);
    static CUdeviceptr gpuAddr(void *addr);
    static CUdeviceptr gpuAddr(const void *addr);

    void memInfo(size_t *free, size_t *total) const;
};

}}

#include "Accelerator.ipp"

#endif
