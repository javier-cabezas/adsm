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

#ifndef __MEMORY_BITMAP_H_
#define __MEMORY_BITMAP_H_

#include <util/Logger.h>

#include <stdint.h>
#include <stddef.h>

namespace gmac { namespace memory  { namespace vm {

class Bitmap : public util::Logger {
private:
#ifdef BITMAP_WORD
    uint32_t *_bitmap;
#else
#ifdef BITMAP_BYTE
    uint8_t *_bitmap;
#else
#ifdef BITMAP_BIT
    uint32_t *_bitmap;
#else
#error "Bitmap granularity not defined"
#endif
#endif
#endif
    bool _dirty;
    bool _synced;

    void *_device;

    size_t _shiftPage;
    size_t _shiftEntry;
    uint32_t _bitMask;
    size_t _size;

    void allocate();

    template <bool check, bool clear>
    bool CheckAndClear(const void *addr);

    off_t offset(const void *addr) const;
public:
    Bitmap(unsigned bits = 32);
    ~Bitmap();

    void *device();
    void *device(const void * addr);
    void *host() const;
    void *host(const void * addr) const;
    const size_t size() const;
    const size_t size(const void * start, size_t size) const;

    const size_t shiftPage() const;
    const size_t shiftEntry() const;

    bool check(const void *);
    bool checkAndClear(const void *);
    void clear(const void *);

    bool clean() const;

    void sync();
    void reset();

    bool synced() const;
    void synced(bool s);
};

}}}

#include "Bitmap.ipp"

#endif


