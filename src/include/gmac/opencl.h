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

/*!
	Adds an argument to be used by the following call to gmacLaunch()
	\param addr Memory address where the param is stored
	\param size Size, in bytes, of the argument
*/
GMAC_API gmacError_t APICALL __oclPushArgumentWithSize(void *addr, size_t size);
#define __oclPushArgument(a) ({ \
        typeof(a) __a = a; \
        __oclPushArgumentWithSize(&(__a), sizeof(__a)); \
    })

/*!
    Configures the next call
    \param workDim
    \param globalWorkOffset
    \param globalWorkSize
    \param localWorkSize
    \return Error code
*/
GMAC_API gmacError_t APICALL __oclConfigureCall(size_t workDim, size_t *globalWorkOffset,
    size_t *globalWorkSize, size_t *localWorkSize);

/**
 * Launches a kernel execution
 * \param k Handler of the kernel to be executed at the GPU
 */
GMAC_API gmacError_t APICALL __oclLaunch(gmacKernel_t k);

/**
 * Prepares the OpenCL code to be used by the applications 
 * \param code Pointer to the NULL-terminated string that contains the code
 * \param flags Compilation flags or NULL
 */
GMAC_API gmacError_t APICALL __oclPrepareCLCode(const char *code, const char *flags = NULL);

/**
 * Prepares the OpenCL binary to be used by the applications 
 * \param binary Pointer to the array that contains the binary code
 * \param size Size in bytes of the array that contains the binary code
 * \param flags Compilation flags or NULL
 */
GMAC_API gmacError_t APICALL __oclPrepareCLBinary(const unsigned char *binary, size_t size, const char *flags = NULL);

#endif /* OPENCL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
