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

#ifndef __MEMOMRY_ROLLINGMANAGER_H_
#define __MEMOMRY_ROLLINGMANAGER_H_

#include "Handler.h"
#include "RollingRegion.h"

#include <kernel/Context.h>

#include <map>
#include <list>

namespace gmac { namespace memory { namespace manager {

class RollingBuffer : protected gmac::util::RWLock {
private:
   std::list<RollingBlock *> _buffer;
   size_t _max;

public:
   RollingBuffer();

   bool overflows();
   size_t inc(size_t n);
   size_t dec(size_t n);
   bool empty();

   void push(RollingBlock *region);
   RollingBlock *pop();
   RollingBlock *front();
   void remove(RollingBlock *region);

   size_t size();
};

class RollingMap : protected std::map<Context *, RollingBuffer *>,
      protected gmac::util::RWLock {
protected:
   RollingBuffer *createBuffer(Context *);
public:
   RollingMap() : RWLock(LockRollingMap) {};
   ~RollingMap();

   RollingBuffer *currentBuffer();
   RollingBuffer *contextBuffer(Context *ctx);

   void remove(RollingBlock *block);
};

class RollingManager : public Handler {
protected:
   size_t lineSize;
   size_t lruDelta;

   RollingMap rollingMap;
   //std::map<Context *, RollingBuffer *> regionRolling;

   util::Lock writeMutex;
   void *writeBuffer;
   size_t writeBufferSize;
   void writeBack();
	void flushToDevice();

   virtual bool read(void *);
   virtual bool write(void *);

#ifdef DEBUG
   void dumpRolling();
#endif

   // Methods used by RollingBlock to request flushing and invalidating
   friend class RollingRegion;
#if 0
   void invalidate(RollingBlock *region);
#endif
   void flush(RollingBlock *region);
   void forceFlush(RollingBlock *region);

   Region * newRegion(void * addr, size_t count, bool shared);
   int defaultProt();

public:
   RollingManager();
   virtual ~RollingManager();
#if 0
   void *malloc(void *addr, size_t size);
   void *globalMalloc(void *addr, size_t size);
   void free(void *addr);
#endif
   void invalidate();
   void invalidate(const RegionSet & regions);
   void flush();
   void flush(const RegionSet & regions);

   void invalidate(const void *addr, size_t size);
   void flush(const void *addr, size_t size);

	void map(Context *, Region *, void *);
    bool touch(Region * r);
};

}}}

#include "RollingManager.ipp"

#endif
