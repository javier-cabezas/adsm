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
#include <util/Private.h>

namespace gmac {

namespace memory { class Map; }

class Context;
class Accelerator;

class Mode {
protected:
    static gmac::util::Private key;

    Accelerator *__acc;
    Context *__context;
    memory::Map *__map;
    unsigned __count;

public:
    Mode(Accelerator *acc);
    ~Mode();

    inline static void init() { key.set(NULL); }
    inline static Mode *current() {
        Mode *mode = static_cast<Mode *>(Mode::key.get());
        if(mode == NULL) mode = proc->create();
        return mode;
    }
    inline static bool hasCurrent() { return key.get() != NULL; }

    inline void inc() { __count++; }
    inline void destroy() { __count--; if(__count == 0) delete this; }
    inline void attach() {
        Mode *mode = static_cast<Mode *>(Mode::key.get());
        if(mode == this) return;
        if(mode != NULL) mode->destroy();
        Mode::key.set(this);
    }
    inline void detach() {
        Mode *mode = static_cast<Mode *>(Mode::key.get());
        if(mode != NULL) mode->destroy();
        Mode::key.set(NULL);
    }

    void switchTo(Accelerator *acc);
    inline Accelerator &accelerator() { return *__acc; }

    inline Context &context() { return *__context; }
    inline const Context &context() const { return *__context; }
    inline memory::Map &map() { return *__map; }
    inline const memory::Map &map() const { return *__map; }
    
	void *translate(void *addr);
	inline const void *translate(const void *addr) {
        return (const void *)translate((void *)addr);
    }
};

}

#endif /* KERNEL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
