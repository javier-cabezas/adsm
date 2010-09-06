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
    static gmac::util::Private key;
    static unsigned next;

    unsigned __id;

    Accelerator *acc;
    Context *context;
    memory::Map *map;
    unsigned count;

    typedef std::map<gmacKernel_t, Kernel *> KernelMap;
    KernelMap kernels;

    virtual void switchIn() = 0;
    virtual void switchOut() = 0;

	gmacError_t __error;
public:
    Mode(Accelerator *acc);
    ~Mode();

    inline static void init() { key.set(NULL); }
    static Mode *current();
    inline static bool hasCurrent() { return key.get() != NULL; }

    inline void inc() { count++; }
    inline void destroy() { count--; if(count == 0) delete this; }

    inline unsigned id() const { return __id; }

    /*! \brief Attaches the execution mode to the current thread */
    void attach();

    /*! \brief Dettaches the execution mode to the current thread */
    void detach();

    inline void addObject(memory::Object *obj) { map->insert(obj); }
#ifndef USE_MMAP
    inline void addReplicatedObject(memory::Object *obj) { map->insertShared(obj); }
    inline void addCentralizedObject(memory::Object *obj) { map->insertGlobal(obj); }
    bool requireUpdate(memory::Block *block);
#endif
    inline void removeObject(memory::Object *obj) { map->remove(obj); }
    inline memory::Object *findObject(const void *addr) {
        return map->find(addr);
    }
    inline const memory::Map &objects() { return *map; }

    /*!  \brief Allocates memory on the accelerator memory */
	virtual gmacError_t malloc(void **addr, size_t size, unsigned align = 1);

	/*!  \brief Releases memory previously allocated by Malloc */
	virtual gmacError_t free(void *addr);

	/*!  \brief Copies data from system memory to accelerator memory */
	virtual gmacError_t copyToDevice(void *dev, const void *host, size_t size);

	/*!  \brief Copies data from accelerator memory to system memory */
	virtual gmacError_t copyToHost(void *host, const void *dev, size_t size);

	/*!  \brief Copies data from accelerator memory to accelerator memory */
	virtual gmacError_t copyDevice(void *dst, const void *src, size_t size);

    /*!  \brief Sets the contents of accelerator memory */
    virtual gmacError_t memset(void *addr, int c, size_t size);

	/*!  \brief Launches the execution of a kernel */
	virtual gmac::KernelLaunch * launch(gmacKernel_t kernel);

	/*!  \brief Waits for kernel execution */
	virtual gmacError_t sync();


    /*!  \brief Returns a buffer to be used by I/O operations */
    virtual IOBuffer *getIOBuffer() = 0;

    virtual gmacError_t bufferToDevice(IOBuffer *buffer, void *dst, size_t size) = 0;
    virtual gmacError_t bufferToHost(IOBuffer *buffer, void *dst, size_t size) = 0;


    void kernel(gmacKernel_t k, Kernel * kernel);
    //Kernel * kernel(gmacKernel_t k);

    
    /*!  \brief Returns the last error code */
    inline gmacError_t error() const { return __error; }

    /*!  \brief Sets up the last error code */
    inline void error(gmacError_t err) { __error = err; }
};

}

#endif /* KERNEL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
