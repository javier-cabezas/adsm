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

#ifndef __API_CUDADRV_CONTEXT_H_
#define __API_CUDADRV_CONTEXT_H_

#include <config.h>
#include <debug.h>
#include <paraver.h>

#include "Accelerator.h"
#include "Kernel.h"
#include "Module.h"

#include "util/Lock.h"
#include "kernel/Context.h"

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>

#include <vector>
#include <map>

namespace gmac { namespace gpu {

class AlignmentMap : public std::map<CUdeviceptr, CUdeviceptr>
{};

class Context : public gmac::Context {
protected:
    AlignmentMap _alignMap;

	Accelerator  &_gpu;
	ModuleVector _modules;

	KernelConfig _call;

    static void * FatBin;
	static const unsigned USleepLaunch = 100;

	typedef std::map<void *, void *> AddressMap;
	static AddressMap hostMem;

#ifdef USE_VM
	static const char *pageTableSymbol;
	const Variable *pageTable;
	struct {
		void *ptr;
		size_t shift;
		size_t size;
		size_t page;
	} devicePageTable;
#elif USE_MMAP
	static const char *baseRegisterSymbol;
#endif

#ifdef USE_MULTI_CONTEXT
	CUcontext _ctx;
	util::Lock mutex;
#endif

    CUstream streamLaunch;
    CUstream streamToDevice;
    CUstream streamToHost;
    CUstream streamDevice;

    bool _pendingKernel;
    bool _pendingToDevice;
    bool _pendingToHost;

    CUdeviceptr gpuAddr(void *addr) const;
	CUdeviceptr gpuAddr(const void *addr) const;
	void zero(void **addr) const;

	friend class Accelerator;

    /*! Auxiliary functions used during Context creation
     */
	void setup();
    void setupStreams();

	Context(Accelerator &gpu);

	~Context();
public:
    static gmacError_t error(CUresult);

    static Context * current();

	void lock();
    void unlock();

    //
	// Standard Accelerator Interface
    //
	gmacError_t malloc(void **addr, size_t size);
	gmacError_t halloc(void **addr, size_t size);
	gmacError_t free(void *addr);

	gmacError_t hostAlloc(void **host, void **device, size_t size);
	gmacError_t hostMemAlign(void **host, void **device, size_t size);
	gmacError_t hostMap(void *host, void **device, size_t size);
	gmacError_t hostFree(void *addr);

	gmacError_t copyToDevice(void *dev, const void *host, size_t size);
	gmacError_t copyToHost(void *host, const void *dev, size_t size);
	gmacError_t copyDevice(void *dst, const void *src, size_t size);

	gmacError_t copyToDeviceAsync(void *dev, const void *host, size_t size);
	gmacError_t copyToHostAsync(void *host, const void *dev, size_t size);

#if 0
    gmacError_t copyDeviceAsync(void *src, const void *dest, size_t size);
#endif

	gmacError_t memset(void *dev, int c, size_t size);
    gmac::KernelLaunch * launch(gmacKernel_t kernel);

    gmacError_t sync();
    gmacError_t syncToHost();
    gmacError_t syncToDevice();
    gmacError_t syncDevice();

    //
	// CUDA-related methods
	//
	const Variable *constant(gmacVariable_t key) const;
    const Variable *variable(gmacVariable_t key) const;
    const Texture  *texture(gmacTexture_t key) const;

	void call(dim3 Dg, dim3 Db, size_t shared, int tokens);
	void argument(const void *arg, size_t size, off_t offset);

    bool async() const;

	void flush(const char * kernel);
	void invalidate();
};

}}

#include "Context.ipp"

#endif
