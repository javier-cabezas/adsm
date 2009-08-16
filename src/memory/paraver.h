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

#ifndef __MEMORY_PARAVER_H
#define __MEMORY_PARAVER_H

#include <paraver.h>

#ifdef PARAVER_GMAC

#include <acc/api.h>

#include <paraver/Trace.h>
#include <paraver/Types.h>

#ifdef __cplusplus
extern "C" {
#endif
	inline gmacError_t __paraver__gmacMalloc(void **devPtr, size_t count) {
		enterFunction(_accMalloc_);
		gmacError_t ret = __gmacMalloc(devPtr, count);
		exitFunction();
		return ret;
	}

	inline gmacError_t __paraver__gmacFree(void *devPtr) {
		enterFunction(_accFree_);
		gmacError_t ret = __gmacFree(devPtr);
		exitFunction();
		return ret;
	}

	inline gmacError_t __paraver__gmacMemcpyToDevice(void *dstPtr, void *srcPtr, size_t count) {
		pushEvent(_HostDeviceCopy_, count);
		enterFunction(_accHostDeviceCopy_);
		gmacError_t ret = __gmacMemcpyToDevice(dstPtr, srcPtr, count);
		exitFunction();
		return ret;
	}

	inline gmacError_t __paraver__gmacMemcpyToHost(void *dstPtr, const void *srcPtr, size_t count) {
		pushEvent(_DeviceHostCopy_, count);
		enterFunction(_accDeviceHostCopy_);
		gmacError_t ret = __gmacMemcpyToHost(dstPtr, srcPtr, count);
		exitFunction();
		return ret;
	}

	inline gmacError_t __paraver__gmacMemcpyDevice(void *dstPtr, const void *srcPtr, size_t count) {
		pushEvent(_DeviceDeviceCopy_, count);
		enterFunction(_accDeviceDeviceCopy_);
		gmacError_t ret = __gmacMemcpyDevice(dstPtr, srcPtr, count);
		exitFunction();
		return ret;
	}

	inline gmacError_t __paraver__gmacMemcpyToDeviceAsync(void *dstPtr, const void *srcPtr, size_t count) {
		pushEvent(_HostDeviceCopy_, count);
		enterFunction(_accHostDeviceCopy_);
		gmacError_t ret = __gmacMemcpyToDeviceAsync(dstPtr, srcPtr, count);
		exitFunction();
		return ret;
	}

	inline gmacError_t __paraver__gmacMemcpyToHostAsync(void *dstPtr, const void *srcPtr, size_t count) {
		pushEvent(_DeviceHostCopy_, count);
		enterFunction(_accDeviceHostCopy_);
		gmacError_t ret = __gmacMemcpyToHostAsync(dstPtr, srcPtr, count);
		exitFunction();
		return ret;
	}

	inline gmacError_t __paraver__gmacLaunch(const char *kernel) {
		pushEvent(_GPUCall_);
		enterFunction(_accLaunch_);
		gmacError_t ret = __gmacLaunch(kernel);
		exitFunction();
		return ret;
	}

	inline gmacError_t __paraver__gmacThreadSynchronize(void) {
		enterFunction(_accSync_);
		gmacError_t ret = __gmacThreadSynchronize();
		exitFunction();
		return ret;
	}
#ifdef __cplusplus
};
#endif

#undef __gmacMalloc
#define __gmacMalloc(...) __paraver__gmacMalloc(__VA_ARGS__)
#undef __gmacFree
#define __gmacFree(...) __paraver__gmacFree(__VA_ARGS__)
#undef __gmacMemcpyToDevice
#define __gmacMemcpyToDevice(...) __paraver__gmacMemcpyToDevice(__VA_ARGS__)
#undef __gmacMemcpyToHost
#define __gmacMemcpyToHost(...) __paraver__gmacMemcpyToHost(__VA_ARGS__)
#undef __gmacMemcpyDevice
#define __gmacMemcpyDevice(...) __paraver__gmacMemcpyDevice(__VA_ARGS__)
#undef __gmacMemcpyToDeviceAsync
#define __gmacMemcpyToDeviceAsync(...) __paraver__gmacMemcpyToDeviceAsync(__VA_ARGS__)
#undef __gmacMemcpyToHostAsync
#define __gmacMemcpyToHostAsync(...) __paraver__gmacMemcpyToHostAsync(__VA_ARGS__)
#undef __gmacLaunch
#define __gmacLaunch(...) __paraver__gmacLaunch(__VA_ARGS__)
#undef __gmacThreadSynchronize
#define __gmacThreadSynchronize(...) __paraver__gmacThreadSynchronize(__VA_ARGS__)

#endif

#endif
