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

/*! \mainpage Global Memory for ACcelerators (GMAC)
 *
 * \section intro_sec Code layout
 * 
 * GMAC is organized in modules. There is an abstract front-end which
 * implements the public API offered to the programmers. These functions
 * use abstract classes that define the Backend API (kernel) and the Memory
 * Management API (memory). Finally, the available backends and memory managers
 * implement the functionality defined in their respective APIs. Currently,
 * the code is organized as follows:
 * \verbatim
 * src/                - GMAC root directory 
 * src/gmac/           - Frontend directory
 * src/core/           - Backend API base classes
 * src/api/            - Backend root directory
 * src/api/cuda        - CUDA run-time backend (no threading support)
 * src/api/cudadrv     - CUDA driver backend (full-featured)
 * src/memory/         - Memory API base classes
 * src/memory/manager  - Memory Managers
 * tests/              - Tests used to validate GMAC
 * \endverbatim
 */


#ifndef GMAC_GMAC_GMAC_H_
#define GMAC_GMAC_GMAC_H_

#include <stddef.h>

// TODO: add define to check for Windows
#include <pthread.h>
typedef pthread_t THREAD_T;

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
    gmacErrorAlreadyBound,
	gmacErrorApiFailureBase,
    gmacErrorFeatureNotSupported,
    gmacErrorInsufficientDeviceMemory,
	gmacErrorUnknown
} gmacError_t;

typedef const char * gmacKernel_t;

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
    "Feature not supported with the current configure configuration",
    "Insufficient memory in the device",
	"Uknown error"
};

/*!
	\brief Returns the number of available accelerators

    This number can be used to perform a manual context distribution among accelerators
*/
size_t gmacAccs();


/*!
	\brief Migrates the GPU execution mode of a thread to a concrete accelerator

    Sets the affinity of a thread to a concrete accelerator. Valid values are 0 ... gmacAccs() - 1.
    Currently only works if this is the first gmac call in the thread.
	\param acc index of the preferred accelerator
*/
gmacError_t gmacMigrate(int acc);


/*!
	\brief Allocates memory at the GPU

	Allocates a range of memory at the GPU and the CPU. Both, GPU and CPU,
	use the same addresses for this memory.
	\param devPtr memory address to store the address for the allocated memory
	\param count  bytes to be allocated
*/
gmacError_t gmacMalloc(void **devPtr, size_t count);


enum GmacGlobalMallocType {
    GMAC_GLOBAL_MALLOC_REPLICATED  = 0,
    GMAC_GLOBAL_MALLOC_CENTRALIZED = 1
};

/*!
	\brief Allocates global memory at all GPUS
*/
#define gmacGlobalMalloc(a, b, ...) __gmacGlobalMalloc(a, b, ##__VA_ARGS__, GMAC_GLOBAL_MALLOC_CENTRALIZED)
gmacError_t __gmacGlobalMalloc(void **devPtr, size_t count, enum GmacGlobalMallocType hint, ...);

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
	\param k Handler of the kernel to be executed at the GPU
*/
gmacError_t gmacLaunch(gmacKernel_t k);
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

void gmacSend(THREAD_T);
void gmacReceive(void);
void gmacSendReceive(THREAD_T);
void gmacCopy(THREAD_T);

#ifdef __cplusplus
#include <cassert>
#include <cstdio>
#else
#include <assert.h>
#include <stdio.h>
#endif

static inline const char *gmacGetErrorString(gmacError_t err) {
	//assert(err <= gmacErrorUnknown);
    if (err <= gmacErrorUnknown)
	return error[err];
    else {
        printf("Error %d\n", err);
        return "WTF Error";
    }
}

#ifdef __cplusplus
};
#endif

#ifdef __cplusplus
template<typename T> inline T *gmacPtr(T *devPtr) {
	return (T *)gmacPtr((void *)devPtr);
}
#endif

#endif
