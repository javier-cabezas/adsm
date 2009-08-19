/* Copyright (c) 2009 University of Illinois
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

#ifndef __API_CUDADRV_GPU_H_
#define __API_CUDADRV_GPU_H_

#include <kernel/Accelerator.h>

#include <cuda.h>
#include <vector_types.h>

namespace gmac {

class GPU : public Accelerator {
protected:
	inline CUdeviceptr gpuAddr(void *addr) const {
		unsigned long a = (unsigned long)addr;
		return (CUdeviceptr)(a & 0xffffffff);
	}

	inline CUdeviceptr gpuAddr(const void *addr) const {
		unsigned long a = (unsigned long)addr;
		return (CUdeviceptr)(a & 0xffffffff);
	}

	unsigned id;
	CUdevice _device;

public:
	GPU(int n, CUdevice device) : id(n), _device(device) {};

	inline CUdevice device() const {
		return _device;
	}

	inline gmacError_t malloc(void **addr, size_t size) {
		*addr = NULL;
		CUresult ret = cuMemAlloc((CUdeviceptr *)addr, size);
		return error(ret);
	}

	inline gmacError_t free(void *addr) {
		CUresult ret = cuMemFree(gpuAddr(addr));
		return error(ret);
	}
		
	inline gmacError_t copyToDevice(void *dev, const void *host, size_t size) {
		CUresult ret = cuMemcpyHtoD(gpuAddr(dev), host, size);
		return error(ret);
	}
	inline gmacError_t copyToHost(void *host, const void *dev, size_t size) {
		CUresult ret = cuMemcpyDtoH(host, gpuAddr(dev), size);
		return error(ret);
	}
	inline gmacError_t copyDevice(void *dst, const void *src, size_t size) {
		CUresult ret = cuMemcpyDtoD(gpuAddr(dst), gpuAddr(src), size);
		return error(ret);
	}
	inline gmacError_t copyToDeviceAsync(void *dev, const void *host,
			size_t size) {
		CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, size, 0);
		return error(ret);
	}
	inline gmacError_t copyToHostAsync(void *host, const void *dev,
			size_t size) {
		CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), size, 0);
		return error(ret);
	}

	gmacError_t memset(void *dev, int value, size_t size);

	gmacError_t launch(dim3, dim3, CUfunction);
	inline gmacError_t sync() {
		CUresult ret = cuCtxSynchronize();
		return error(ret);
	}

	gmacError_t error(CUresult r);

};

}

#endif
