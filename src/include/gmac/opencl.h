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
typedef gmac_kernel_id_t ocl_kernel_id_t;
typedef enum GmacGlobalMallocType OclMemoryHint;
typedef enum GmacProtection OclProtection;

#define OCL_GLOBAL_MALLOC_CENTRALIZED GMAC_GLOBAL_MALLOC_CENTRALIZED
#define OCL_GLOBAL_MALLOC_DISTRIBUTED GMAC_GLOBAL_MALOOC_DISTRIBUTED

/*!
 *  Adds an argument to be used by the following call to __oclLaunch()
 *  \param index Index of the parameter being added in the parameter list
 *  \return Error code
*/
GMAC_API oclError_t APICALL __oclKernelGet(ocl_kernel_id_t id, OclKernel *kernel);


/**
 *  Adds an argument to be used by the following call to __oclLaunch()
 *  \param kernel Kernel descriptor
 *  \param addr Memory address where the param is stored
 *  \param size Size, in bytes, of the argument
 *  \param index Index of the parameter being added in the parameter list
 *  \return Error code
 */
GMAC_API oclError_t APICALL __oclKernelSetArg(OclKernel *kernel, const void *addr, size_t size, unsigned index);

/**
 *  Configures the next call
 *  \param kernel Kernel descriptor
 *  \param workDim
 *  \param globalWorkOffset
 *  \param globalWorkSize
 *  \param localWorkSize
 *  \return Error code
 */
GMAC_API oclError_t APICALL __oclKernelConfigure(OclKernel *kernel, size_t workDim, size_t *globalWorkOffset,
    size_t *globalWorkSize, size_t *localWorkSize);

/**
 * Launches a kernel execution
 * \param kernel Handler of the kernel to be executed at the GPU
 * \return Error code
 */
GMAC_API oclError_t APICALL __oclKernelLaunch(OclKernel *kernel);

/**
 * Waits for kernel execution finalization
 * \param kernel Handler of the kernel to wait for
 * \return Error code
 */
GMAC_API oclError_t APICALL __oclKernelWait(OclKernel *kernel);

/**
 * Launches a kernel execution
 * \param kernel Handler of the kernel to be executed at the GPU
 * \return Error code
 */
GMAC_API oclError_t APICALL __oclKernelDestroy(OclKernel *kernel);

/**
 * Prepares the OpenCL code to be used by the application
 * \param code Pointer to the NULL-terminated string that contains the code
 * \param flags Compilation flags or NULL
 */
GMAC_API oclError_t APICALL __oclPrepareCLCode(const char *code, const char *flags = NULL);

/**
 * Prepares the OpenCL code in the specified fie to be used by the application
 * \param path String pointing to the file with the code to be added
 * \param flags Compilation flags or NULL
 */
GMAC_API oclError_t APICALL __oclPrepareCLCodeFile(const char *path, const char *flags = NULL);

/**
 * Prepares the OpenCL binary to be used by the application
 * \param binary Pointer to the array that contains the binary code
 * \param size Size in bytes of the array that contains the binary code
 * \param flags Compilation flags or NULL
 */
GMAC_API oclError_t APICALL __oclPrepareCLBinary(const unsigned char *binary, size_t size, const char *flags = NULL);


/* Wrappers to GMAC native calls */
/** Get the number of accelerators in the system
 * \return Number of accelerators
 */
static inline
unsigned oclGetNumberOfAccelerators() { return gmacGetNumberOfAccelerators(); }

/** Get the amount of available accelerator memory
 * \return Size (in bytes) of the available accelerator memory
 */
static inline
size_t oclGetFreeMemory() { return gmacGetFreeMemory(); }

/** Attach the calling CPU thread to a different accelerator
 * \param acc Accelerator to attach the current CPU thread
 * \return Error code
 */
static inline
oclError_t oclMigrate(unsigned acc) { return gmacMigrate(acc); }

/** Map host memory in the accelerator
 * \param cpuPtr Host memory address to map
 * \param count Size (in bytes) to be mapped in accelerator memory
 * \param prot Desired memory protection of the mapping
 * \return Error code
 */
static inline
oclError_t oclMemoryMap(void *cpuPtr, size_t count, OclProtection prot) {
    return gmacMemoryMap(cpuPtr, count, prot);
}

/** Unmap host memory from the accelerator
 * \param cpuPtr Host memory address to be unmmaped
 * \param count Size (in bytes) to be unmmaped
 * \return Error code
 */
static inline
oclError_t oclMemoryUnmap(void *cpuPtr, size_t count) { return gmacMemoryUnmap(cpuPtr, count); }

/** Allocate shared memory
 * \param devPtr Memory address of the pointer to store the allocated memory
 * \param count Size (in bytes) of the memory to be allocated
 * \return Error code
 */
static inline
oclError_t oclMalloc(void **devPtr, size_t count) { return gmacMalloc(devPtr, count); }

/** Allocate shared memory accessible from all accelerators
 * \param devPtr Memory address of the pointer to store the allocated memory
 * \param count Size (in bytes) of the memory to be allocated
 * \param hint Type of desired global memory
 * \return Error code
 */
static inline
oclError_t oclGlobalMalloc(void **devPtr, size_t count, OclMemoryHint hint __dv(OCL_GLOBAL_MALLOC_CENTRALIZED)) {
    return gmacGlobalMalloc(devPtr, count, hint);
}

/** Get the OpenCL memory object associated to a shared memory address
 * \param cpuPtr Host shared memory address
 * \return Associated OpenCL buffer
 */
static inline
cl_mem oclPtr(const void *cpuPtr) { return gmacPtr(cpuPtr); }

/** Release shared memory
 * \param cpuPtr Shared memory address to be released
 * \return Error code
 */
static inline
oclError_t oclFree(void *cpuPtr) { return gmacFree(cpuPtr); }

/** Wait until all previous accelerator calls are completed
 * \return Error code
 */
static inline
oclError_t oclThreadSynchronize() { return gmacThreadSynchronize(); }

/** Get the last error produced by GMAC
 * \return Error code
 */
static inline
oclError_t oclGetLastError() { return gmacGetLastError(); }

/** Initialize a shared memory region
 * \param cpuPtr Starting shared memory address 
 * \param c Value used to be initialized
 * \param count Size (in bytes) of the shared memory region to be initialized
 * \return Shared memory address that has been initialized
 */
static inline
void *oclMemset(void *cpuPtr, int c, size_t count) { return gmacMemset(cpuPtr, c, count); }

/** Copy data between shared memory regions
 * \param cpuDstPtr Destination shared memory
 * \param cpuSrcPtr Source shared memory
 * \param count Size (in bytes) to be copied
 * \return Destination shared memory address
 */
static inline
void *oclMemcpy(void *cpuDstPtr, const void *cpuSrcPtr, size_t count) {
    return gmacMemcpy(cpuDstPtr, cpuSrcPtr, count);
}

/** Send the execution mode associated to the current CPU thread to another CPU thread
 * \param tid Thread ID of the destionation CPU thread
 */
static inline
void oclSend(THREAD_T tid) { return gmacSend(tid); }

/** Receive an execution mode from another CPU thread */
static inline
void oclReceive(void) { return gmacReceive(); }

/** Send the execution mode associated to the current CPU thread and wait to receive a new execution mode
 * \param tid Thread ID of the destination CPU thread
 */
static inline
void oclSendReceive(THREAD_T tid) { return gmacSendReceive(tid); }

/** Create a copy of the execution mode associate to the current CPU thread and send that copy another CPU thread
 * \param tid Thread ID of the destination CPU thread
 */
static inline
void oclCopy(THREAD_T tid) { return gmacCopy(tid); }

#ifdef __cplusplus
}

/** Get the OpenCL memory object associated to a shared memory address
 * \param cpuPtr Host shared memory address
 * \return Associated OpenCL buffer
 */
template<typename T>
static inline cl_mem oclPtr(const T *addr) {
    return gmacPtr((const void *)addr);
}

#endif

#undef __dv

#endif /* OPENCL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
