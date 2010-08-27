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

#ifndef __API_CUDA_MODE_H_
#define __API_CUDA_MODE_H_

#include <config.h>
#include <paraver.h>

#include <kernel/Mode.h>

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>


namespace gmac { namespace gpu {

class Buffer : public util::Lock {
protected:
    void *__buffer;
    size_t __size;
    bool __ready;
    friend class Mode;
public:
    Buffer(paraver::LockName name, Mode *mode);
    ~Buffer();

    inline void *ptr() const { return __buffer; }
    inline size_t size() const { return __size; }
    inline bool ready() const { return __ready; }
    inline void busy() { __ready = false; }
    inline void idle() { __ready = true; }
};

#ifdef USE_MULTI_CONTEXT
class ContextLock : public util::Lock {
protected:
    friend class Context;
public:
    Context() : util::Lock() {};
};
#endif


class Mode : public gmac::Mode {
protected:
    CUstream __exe;
    CUstream __host;
    CUstream __device;
    CUstream __internal;

    Buffer __hostBuffer;
    Buffer __deviceBuffer;

#ifdef USE_MULTI_CONTEXT
    CUcontext __ctx;
    ContextLock __mutex;
#endif

    virtual void switchIn();
    virtual void switchOut();

    void syncStream(CUstream stream);
public:
    Mode(Accelerator *acc);
    virtual ~Mode();

	virtual gmacError_t copyToDevice(void *dev, const void *host, size_t size);
	virtual gmacError_t copyToHost(void *host, const void *dev, size_t size);
	virtual gmacError_t copyDevice(void *dst, const void *src, size_t size);

    virtual gmacError_t sync();
    virtual gmac::KernelLaunch launch(const char *);

    gmacError_t hostAlloc(void **addr, size_t size);
    gmacError_t hostFree(void *addr);
};

}}

#include "Mode.ipp"

#endif
