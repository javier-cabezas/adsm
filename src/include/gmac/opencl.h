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

#ifndef GMAC_OPENCL_H_
#define GMAC_OPENCL_H_

#include "CL/cl.h"
#include "opencl_types.h"

#include "api.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#   define __dv(a) = a
#else
#   define __dv(a)
#endif

typedef gmacError_t oclError_t;
typedef enum GmacGlobalMallocType OclMemoryHint;

#define OCL_GLOBAL_MALLOC_CENTRALIZED GMAC_GLOBAL_MALLOC_CENTRALIZED
#define OCL_GLOBAL_MALLOC_DISTRIBUTED GMAC_GLOBAL_MALOOC_DISTRIBUTED

/*!
	Adds an argument to be used by the following call to gmacLaunch()
	\param addr Memory address where the param is stored
	\param size Size, in bytes, of the argument
	\param index Index of the parameter being added in the parameter list
*/
GMAC_API oclError_t APICALL __oclSetArgument(const void *addr, size_t size, unsigned index);

/*!
    Configures the next call
    \param workDim
    \param globalWorkOffset
    \param globalWorkSize
    \param localWorkSize
    \return Error code
*/
GMAC_API oclError_t APICALL __oclConfigureCall(size_t workDim, size_t *globalWorkOffset,
    size_t *globalWorkSize, size_t *localWorkSize);

/**
 * Launches a kernel execution
 * \param k Handler of the kernel to be executed at the GPU
 */
GMAC_API oclError_t APICALL __oclLaunch(gmacKernel_t k);

/**
 * Prepares the OpenCL code to be used by the applications 
 * \param code Pointer to the NULL-terminated string that contains the code
 * \param flags Compilation flags or NULL
 */
GMAC_API oclError_t APICALL __oclPrepareCLCode(const char *code, const char *flags = NULL);

/**
 * Prepares the OpenCL binary to be used by the applications 
 * \param binary Pointer to the array that contains the binary code
 * \param size Size in bytes of the array that contains the binary code
 * \param flags Compilation flags or NULL
 */
GMAC_API oclError_t APICALL __oclPrepareCLBinary(const unsigned char *binary, size_t size, const char *flags = NULL);


/* Wrappers to GMAC native calls */
static inline
unsigned oclGetNumberOfAccelerators() { return gmacGetNumberOfAccelerators(); }

static inline
size_t oclGetFreeMemory() { return gmacGetFreeMemory(); }

static inline
oclError_t oclMigrate(unsigned acc) { return gmacMigrate(acc); }

static inline
oclError_t oclMemoryMap(void *cpuPtr, size_t count, enum GmacProtection prot) {
    return gmacMemoryMap(cpuPtr, count, prot);
}

static inline
oclError_t oclMemoryUnmap(void *cpuPtr, size_t count) { return gmacMemoryUnmap(cpuPtr, count); }

static inline
oclError_t oclMalloc(void **devPtr, size_t count) { return gmacMalloc(devPtr, count); }

static inline
oclError_t oclGlobalMalloc(void **devPtr, size_t count, OclMemoryHint hint __dv(OCL_GLOBAL_MALLOC_CENTRALIZED)) {
    return gmacGlobalMalloc(devPtr, count, hint);
}

static inline
cl_mem oclPtr(const void *cpuPtr) { return gmacPtr(cpuPtr); }

static inline
oclError_t oclFree(void *cpuPtr) { return gmacFree(cpuPtr); }

static inline
oclError_t oclThreadSynchronize() { return gmacThreadSynchronize(); }

static inline
oclError_t oclGetLastError() { return gmacGetLastError(); }

static inline
void *oclMemset(void *cpuPtr, int c, size_t count) { return gmacMemset(cpuPtr, c, count); }

static inline
void *oclMemcpy(void *cpuDstPtr, const void *cpuSrcPtr, size_t count) {
    return gmacMemcpy(cpuDstPtr, cpuSrcPtr, count);
}

static inline
void oclSend(THREAD_T tid) { return gmacSend(tid); }

static inline
void oclReceive(void) { return gmacReceive(); }

static inline
void oclSendReceive(THREAD_T tid) { return gmacSendReceive(tid); }

static inline
void oclCopy(THREAD_T tid) { return gmacCopy(tid); }

#ifdef __cplusplus
}
#endif

template<typename T>
static inline cl_mem oclPtr(const T *addr) {
    return gmacPtr((const void *)addr);
}

#undef __dv

#endif /* OPENCL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
