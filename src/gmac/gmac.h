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

#ifndef __GMAC_H_
#define __GMAC_H_

#include <threads.h>

#include <assert.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	gmacSuccess = 0,
	gmacErrorMemoryAllocation,
	gmacErrorLaunchFailure,
	gmacErrorNotReady,
	gmacErrorNoDevice,
	gmacErrorInvalidValue,
	gmacErrorInvalidDevice,
	gmacErrorInvalidDeviceFunction,
	gmacErrorApiFailureBase,
	gmacErrorUnknown
} gmacError_t;

static const char *error[] = {
	"No error",
	"Memory allocation",
	"Launch failure",
	"Device is not ready",
	"Device is not present",
	"Invalid value",
	"Invalid device",
	"Invalid device function",
	"GMAC general failure",
	"Uknown error"
};

/*!
	\brief Allocates memory at the GPU
	
	Allocates a range of memory at the GPU and the CPU. Both, GPU and CPU,
	use the same addresses for this memory.
	\param devPtr memory address to store the address for the allocated memory
	\param count bytes to be allocated
*/
gmacError_t gmacMalloc(void **devPtr, size_t count);

/*
	\brief Allocates global memory at all GPUS
*/
gmacError_t gmacGlobalMalloc(void **devPtr, size_t count);

/*!
	\brief Gets a GPU address

	\param cpuPtr memory address at the CPU
*/
void *gmacPtr(void *cpuPtr);
/*!
	\brief Free the memory allocated with gmacMalloc() and gmacSafeMalloc()
	\param cpuAddr Memory address to free. This address must have been returned
	by a previous call to gmacMalloc() or gmacSafeMalloc()
*/
gmacError_t gmacFree(void *);
/*!
	\brief Launches a kernel execution
	\param name Name of the kernel to be executed at the GPU
*/
gmacError_t gmacLaunch(const char *name);
/*!
	\brief Waits until all previous GPU requests have finished
*/
gmacError_t gmacThreadSynchronize(void);
/*!
	\brief Sets up an argument to be used by the following call to gmacLaunch()
	\param addr Memory address where the param is stored
	\param size Size, in bytes, of the argument
	\param offset Offset, in bytes, of the argument in the argument list
*/
gmacError_t gmacSetupArgument(void *addr, size_t size, size_t offset);

gmacError_t gmacGetLastError(void);

void *gmacMemset(void *, int, size_t);
void *gmacMemcpy(void *, const void *, size_t);

void gmacSendReceive(THREAD_ID);

inline const char *gmacGetErrorString(gmacError_t err) {
	assert(err <= gmacErrorUnknown);
	return error[err];
}

#ifdef __cplusplus
};
#endif

#ifdef __cplusplus
#include <stdio.h>
template<typename T> inline T *gmacPtr(T *devPtr) {
	return (T *)gmacPtr((void *)devPtr);
}
#endif

#endif
