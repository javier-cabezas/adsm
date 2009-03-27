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

#include <cuda_runtime.h>
#include <driver_types.h>

typedef cudaError_t gmacError_t;
typedef enum cudaMemcpyKind gmacMemcpyKind;
typedef cudaStream_t gmacStream_t;

void gmacCreateManager(void);
void gmacRemoveManager(void);

extern gmacError_t (*__gmacLaunch)(const char *);

#define __gmacMalloc(...) cudaMalloc(__VA_ARGS__)
#define __gmacMallocPitch(...) cudaMallocPitch(__VA_ARGS__)
#define __gmacFree(...) cudaFree(__VA_ARGS__)
#define __gmacMemcpyToDevice(...) cudaMemcpy(__VA_ARGS__, cudaMemcpyHostToDevice)
#define __gmacMemcpyToHost(...) cudaMemcpy(__VA_ARGS__, cudaMemcpyDeviceToHost)
#define __gmacMemcpyToDeviceAsync(...) cudaMemcpyAsync(__VA_ARGS__, cudaMemcpyHostToDevice, 0)
#define __gmacThreadSynchronize() cudaThreadSynchronize()

#define gmacGetErrorString cudaGetErrorString
#define gmacGetLastError cudaGetLastError

#define gmacSuccess cudaSuccess
#define gmacErrorMemoryAllocation cudaErrorMemoryAllocation

#define gmacMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gmacMemcpyHostToDevice cudaMemcpyHostToDevice

#endif
