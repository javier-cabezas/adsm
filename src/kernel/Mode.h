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

#ifndef __KERNEL_MODE_H
#define __KERNEL_MODE_H

#ifdef USE_VM
#include "memory/Bitmap.h"
#endif

#include <kernel/Process.h>
#include <kernel/Accelerator.h>
#include <kernel/Context.h>

#include <util/Private.h>
#include <util/Logger.h>

namespace gmac {

namespace memory { class Map; class Object; class Block; }

class Context;
class IOBuffer;
class KernelLaunch;

class Mode : public gmac::util::Logger {
protected:
    static gmac::util::Private<Mode> key;
    static unsigned next;

    unsigned _id;

    Accelerator *_acc;
    Context *_context;
    memory::Map *_map;
#ifdef USE_VM
    memory::vm::Bitmap *_bitmap;
#endif
    unsigned _count;

    typedef std::map<gmacKernel_t, Kernel *> KernelMap;
    KernelMap kernels;

    virtual void switchIn() = 0;
    virtual void switchOut() = 0;

	gmacError_t _error;
public:
    Mode(Accelerator *_acc);
    ~Mode();
    void release();

    static void init();
    static void initThread();
    static Mode *current();
    static bool hasCurrent();

    void inc();
    void destroy();

    unsigned id() const;
    unsigned accId() const;

    /*! \brief Attaches the execution mode to the current thread */
    void attach();

    /*! \brief Dettaches the execution mode to the current thread */
    void detach();

    void addObject(memory::Object *obj);
#ifndef USE_MMAP
    void addReplicatedObject(memory::Object *obj);
    void addCentralizedObject(memory::Object *obj);
    bool requireUpdate(memory::Block *block);
#endif
    void removeObject(memory::Object *obj);
    memory::Object *findObject(const void *addr);
    const memory::Map &objects();

    /*!  \brief Allocates memory on the accelerator memory */
	gmacError_t malloc(void **addr, size_t size, unsigned align = 1);

	/*!  \brief Releases memory previously allocated by malloc */
	gmacError_t free(void *addr);

	/*!  \brief Copies data from system memory to accelerator memory */
	gmacError_t copyToDevice(void *dev, const void *host, size_t size);

	/*!  \brief Copies data from accelerator memory to system memory */
	gmacError_t copyToHost(void *host, const void *dev, size_t size);

	/*!  \brief Copies data from accelerator memory to accelerator memory */
	gmacError_t copyDevice(void *dst, const void *src, size_t size);

    /*!  \brief Sets the contents of accelerator memory */
    gmacError_t memset(void *addr, int c, size_t size);

	/*!  \brief Launches the execution of a kernel */
	KernelLaunch * launch(gmacKernel_t kernel);
	virtual gmacError_t execute(KernelLaunch * launch) = 0;

	/*!  \brief Waits for kernel execution */
	gmacError_t sync();


    virtual gmacError_t bufferToDevice(void *dst, IOBuffer *buffer, size_t size, off_t off = 0) = 0;
    virtual gmacError_t deviceToBuffer(IOBuffer *buffer, const void *dst, size_t size, off_t off = 0) = 0;

    void kernel(gmacKernel_t k, Kernel * kernel);
    //Kernel * kernel(gmacKernel_t k);

    
    /*!  \brief Returns the last error code */
    gmacError_t error() const;

    /*!  \brief Sets up the last error code */
    void error(gmacError_t err);

#ifdef USE_VM
    memory::vm::Bitmap & dirtyBitmap();
    const memory::vm::Bitmap & dirtyBitmap() const;
#endif
};

}

#include "Mode.ipp"

#endif /* KERNEL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
