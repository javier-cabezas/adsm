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

#include <util/Private.h>
#include <util/Logger.h>

namespace gmac {

namespace memory { class Map; class Object; }

class Context;
class KernelLaunch;

class Mode : public gmac::util::Logger {
protected:
    static gmac::util::Private key;
    static unsigned next;

    unsigned __id;

    Accelerator *__acc;
    Context *__context;
    memory::Map *__map;
    unsigned __count;

    virtual Context *createContext();
    virtual void destroyContext(Context *ctx);

    virtual void switchIn() = 0;
    virtual void switchOut() = 0;

	gmacError_t __error;
public:
    Mode(Accelerator *acc);
    ~Mode();

    inline static void init() { key.set(NULL); }
    inline static Mode *current();
    inline static bool hasCurrent() { return key.get() != NULL; }

    inline void inc() { __count++; }
    inline void destroy() { __count--; if(__count == 0) delete this; }

    inline unsigned id() const { return __id; }

    /*! \brief Attaches the execution mode to the current thread */
    inline void attach();

    /*! \brief Dettaches the execution mode to the current thread */
    inline void detach();

    inline void addObject(memory::Object *obj) { __map->insert(obj); }
    inline void removeObject(memory::Object *obj) { __map->remove(obj); }
    inline memory::Object *findObject(const void *addr) {
        return __map->find(addr);
    }

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

	/*!  \brief Waits for kernel execution */
	virtual gmacError_t sync() = 0;

	/*!  \brief Launches the execution of a kernel */
	virtual gmac::KernelLaunch * launch(gmacKernel_t kernel) = 0;


    inline gmacError_t error() const { return __error; }
};

}

#include "Mode.ipp"

#endif /* KERNEL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
