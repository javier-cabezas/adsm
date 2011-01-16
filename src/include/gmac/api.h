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

/** \mainpage Global Memory for ACcelerators (GMAC)
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
 * src/include/        - Instalable include files
 * src/core/           - Backend API base classes
 * src/api/            - Backend root directory
 * src/api/cuda        - CUDA run-time backend
 * src/api/opencl      - OpenCL run-time backend
 * src/memory/         - Memory API base classes
 * src/memory/manager  - Memory Managers
 * tests/              - Tests used to validate GMAC
 * \endverbatim
 */

#ifndef GMAC_API_H_
#define GMAC_API_H_

#include <stddef.h>

#include "types.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Returns the number of available accelerators.
 * This number can be used to perform a manual context distribution among
 * accelerators
 */
GMAC_API size_t APICALL gmacAccs();


/**
 * Migrates the GPU execution mode of a thread to a concrete accelerator.
 * Sets the affinity of a thread to a concrete accelerator. Valid values are 0
 * ... gmacAccs() - 1.
 * Currently only works if this is the first gmac call in the thread.
 *
 * \param acc index of the preferred accelerator
 */
GMAC_API gmacError_t APICALL gmacMigrate(int acc);

/**
 * Maps a range of CPU memory on the GPU. Both, GPU and CPU,
 * use the same addresses for this memory.
 *
 * \param cpuPtr CPU memory address to be mapped on the GPU
 * \param count Number of bytes to be allocated
 * \param prot The protection to be used in the mapping (currently unused)
 */
GMAC_API gmacError_t APICALL gmacMap(void *cpuPtr, size_t count, enum GmacProtection prot);

/*!
	Unmaps a range of CPU memory from the GPU. Both, GPU and CPU,
	use the same addresses for this memory.
	\param cpuPtr memory address to be unmapped from the GPU
	\param count  bytes to be allocated
*/
GMAC_API gmacError_t APICALL gmacUnmap(void *cpuPtr, size_t count);

/*!
	Allocates a range of memory in the GPU and the CPU. Both, GPU and CPU,
	use the same addresses for this memory.
	\param devPtr memory address to store the address for the allocated memory
	\param count  bytes to be allocated
*/
GMAC_API gmacError_t APICALL gmacMalloc(void **devPtr, size_t count);


/*!
	Allocates global memory at all GPUS
*/
#define gmacGlobalMalloc(a, b, ...) __gmacGlobalMalloc(a, b, ##__VA_ARGS__, GMAC_GLOBAL_MALLOC_CENTRALIZED)
GMAC_API gmacError_t APICALL __gmacGlobalMalloc(void **devPtr, size_t count, enum GmacGlobalMallocType hint, ...);

GMAC_API gmacError_t APICALL gmacLaunch(gmacKernel_t k);

/*!
	Gets a GPU address
	\param cpuPtr memory address at the CPU
*/
GMAC_API __gmac_accptr_t APICALL gmacPtr(const void *cpuPtr);

/*!
	Free the memory allocated with gmacMalloc() and gmacSafeMalloc()
	\param cpuPtr Memory address to free. This address must have been returned
	by a previous call to gmacMalloc() or gmacSafeMalloc()
*/
GMAC_API gmacError_t APICALL gmacFree(void *cpuPtr);

/*!
	Waits until all previous GPU requests have finished
*/
GMAC_API gmacError_t APICALL gmacThreadSynchronize(void);


GMAC_API gmacError_t APICALL gmacGetLastError(void);

GMAC_API void * APICALL gmacMemset(void *, int, size_t);
GMAC_API void * APICALL gmacMemcpy(void *, const void *, size_t);

GMAC_API void APICALL gmacSend(THREAD_T);
GMAC_API void APICALL gmacReceive(void);
GMAC_API void APICALL gmacSendReceive(THREAD_T);
GMAC_API void APICALL gmacCopy(THREAD_T);

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


#endif
