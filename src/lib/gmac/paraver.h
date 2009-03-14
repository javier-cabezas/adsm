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

#ifndef __GMAC_PARAVER_H
#define __GMAC_PARAVER_H

#ifdef PARAVER

#include <gmac.h>
#include <paraver/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

void addThread(void);
void pushState(paraver::StateName &s);
void popState(void);
void pushEvent(paraver::EventName &s);

inline cudaError_t _gmacMalloc(void **devPtr, size_t count) {
	pushState(paraver::_gmacMalloc_);
	cudaError_t ret = gmacMalloc(devPtr, count);
	popState();
	return ret;
}

inline cudaError_t _gmacSafeMalloc(void **devPtr, size_t count) {
	pushState(paraver::_gmacMalloc_);
	cudaError_t ret = gmacSafeMalloc(devPtr, count);
	popState();
	return ret;
}

inline cudaError_t _gmacFree(void *devPtr) {
	pushState(paraver::_gmacFree_);
	cudaError_t ret = gmacFree(devPtr);
	popState();
	return ret;
}

inline cudaError_t _gmacLaunch(const char *kernel) {
	pushState(paraver::_gmacLaunch_);
	cudaError_t ret = gmacLaunch(kernel);
	popState();
	return ret;
}

inline cudaError_t _gmacThreadSynchronize(void) {
	pushState(paraver::_gmacSync_);
	cudaError_t ret = gmacThreadSynchronize();
	popState();
	return ret;
}

#ifdef __cplusplus
};
#endif

#define gmacMalloc(...) _gmacMalloc(__VA_ARGS__)
#define gmacSafeMalloc(...) _gmacSafeMalloc(__VA_ARGS__)
#define gmacFree(...) _gmacFree(__VA_ARGS__)
#define gmacLaunch(...) _gmacLaunch(__VA_ARGS__)
#define gmacThreadSynchronize(...) _gmacThreadSynchronize(__VA_ARGS__)

#endif

#endif
