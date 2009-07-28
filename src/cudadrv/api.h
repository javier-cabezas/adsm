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

#ifndef __CUDA_API_H_
#define __CUDA_API_H_

#include <cuda.h>
#include <driver_types.h>

typedef cudaError_t gmacError_t;
typedef enum {
	gmacMemcpyDeviceToHost,
	gmacMemcpyHostToDevice
} gmacMemcpyKind;


void gmacCreateManager(void);
void gmacRemoveManager(void);

gmacError_t __gmacMalloc(void **, size_t);
gmacError_t __gmacMallocPitch(void **, size_t *, size_t, size_t);
gmacError_t __gmacFree(void *);

gmacError_t __gmacMemcpyToDevice(void *, const void *, size_t);
gmacError_t __gmacMemcpyToHost(void *, const void *, size_t);
gmacError_t __gmacMemcpyDevice(void *, const void *, size_t);
gmacError_t __gmacMemcpyToDeviceAsync(void *, const void *, size_t);
gmacError_t __gmacMemcpyToHostAsync(void *, const void *, size_t);

gmacError_t __gmacMemset(void *, int, size_t);

gmacError_t __gmacLaunch(const char *);
gmacError_t __gmacThreadSynchronize(void);

#ifdef __cplusplus
extern "C" {
#endif
gmacError_t gmacGetLastError(void);
const char *gmacGetErrorString(gmacError_t);
#ifdef __cplusplus
}
#endif

gmacError_t __gmacError(CUresult);

#define gmacSuccess cudaSuccess
#define gmacErrorMemoryAllocation cudaErrorMemoryAllocation
#define gmacErrorInvalidDeviceFunction cudaErrorInvalidDeviceFunction
#define gmacErrorLaunchFailure cudaErrorLaunchFailure
#define gmacErrorUnknown cudaErrorUnknown

#endif
