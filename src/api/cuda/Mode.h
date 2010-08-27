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

#include "Kernel.h"
#include "Module.h"

#include <kernel/Mode.h>

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>


namespace gmac { namespace gpu {

class Switch {
public:
    static void in();
    static void out();
};

class Mode;
class Buffer : public util::Logger, public util::Lock {
protected:
    Mode *__mode;

    void *__buffer;
    size_t __size;
    bool __ready;

    friend class Mode;
public:
    Buffer(paraver::LockName name, Mode *__mode);
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
    ContextLock() : util::Lock() {};
};
#endif

class Texture;

class Accelerator;
class Mode : public gmac::Mode {
protected:
    Accelerator *__acc;

    typedef CUstream Stream;
    Stream streamLaunch;
    Stream streamToDevice;
    Stream streamToHost;
    Stream streamDevice;

    void setupStreams();
    void cleanStreams();
    gmacError_t syncStream(CUstream);

    Buffer hostBuffer;
    Buffer deviceBuffer;

    KernelConfig __call;

#ifdef USE_MULTI_CONTEXT
    CUcontext __ctx;
    ContextLock __mutex;
#endif

    friend class Switch;
    virtual void switchIn();
    virtual void switchOut();

public:
    Mode(Accelerator *acc);
    ~Mode();

	gmacError_t copyToDevice(void *dev, const void *host, size_t size);
	gmacError_t copyToHost(void *host, const void *dev, size_t size);
	gmacError_t copyDevice(void *dst, const void *src, size_t size);

    gmacError_t sync();
    gmac::KernelLaunch *launch(const char *);

    gmacError_t hostAlloc(void **addr, size_t size);
    gmacError_t hostFree(void *addr);

    void call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens);
	void argument(const void *arg, size_t size, off_t offset);

    const Variable *constant(gmacVariable_t key) const;
    const Variable *variable(gmacVariable_t key) const;
    const Texture *texture(gmacTexture_t key) const;

    inline Stream eventStream() const { return streamLaunch; }
};

}}

#include "Mode.ipp"

#endif
