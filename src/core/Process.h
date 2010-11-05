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

#ifndef GMAC_CORE_PROCESS_H_
#define GMAC_CORE_PROCESS_H_

#include <list>
#include <map>
#include <vector>

#include "config/common.h"
#include "include/gmac/types.h"
#include "memory/Map.h"
#include "util/Logger.h"
#include "util/Singleton.h"

#include "Queue.h"
#include "allocator/Buddy.h"


namespace gmac {
class Accelerator;
class IOBuffer;
class Mode;
class Context;
class Process;

void apiInit(void);
void contextInit(void);
void memoryInit(const char *manager = NULL, const char *allocator = NULL);
void memoryFini(void);
}

namespace gmac {

namespace memory { class DistributedObject; }


class GMAC_LOCAL ModeMap : private std::map<Mode *, Accelerator *>, public util::RWLock
{
private:
    typedef std::map<Mode *, Accelerator *> Parent;

    friend class Process;
public:
    ModeMap();

    typedef Parent::iterator iterator;
    typedef Parent::const_iterator const_iterator;

    std::pair<iterator, bool> insert(Mode *, Accelerator *);
    size_t remove(Mode &mode);
};

class GMAC_LOCAL QueueMap : private std::map<THREAD_T, ThreadQueue *>, public util::RWLock
{
private:
    typedef std::map<THREAD_T, ThreadQueue *> Parent;
public:
    QueueMap();

    typedef Parent::iterator iterator;

    void cleanup();
    std::pair<iterator, bool> insert(THREAD_T, ThreadQueue *);
    void push(THREAD_T id, Mode &mode);
    void attach();
    void erase(THREAD_T id);
};

class GMAC_LOCAL Process : public util::Singleton<Process>, public util::RWLock, public util::Logger {
	// Needed to let Singleton call the protected constructor
	friend class util::Singleton<Process>;
	//friend class Accelerator;
protected:
    std::vector<Accelerator *> accs_;
    ModeMap modes_;

    QueueMap queues_;
    memory::ObjectMap shared_;
    memory::ObjectMap centralized_;
    memory::ObjectMap replicated_;
    memory::ObjectMap orphans_;

    unsigned current_;

    static size_t TotalMemory_;

    Process();
    core::allocator::Buddy *ioMemory_;

public:
    virtual ~Process();

    void initThread();
    void finiThread();
#define ACC_AUTO_BIND -1
    Mode * createMode(int acc = ACC_AUTO_BIND);
    void removeMode(Mode &mode);

#ifndef USE_MMAP
    gmacError_t globalMalloc(memory::DistributedObject &object, size_t size);
    gmacError_t globalFree(memory::DistributedObject &object);
#endif

    IOBuffer *createIOBuffer(size_t size);
    void destroyIOBuffer(IOBuffer *buffer);

    void *translate(void *addr);
    inline const void *translate(const void *addr) {
        return (const void *)translate((void *)addr);
    }

    /* Context management functions */
    void send(THREAD_T id);
    void receive();
    void sendReceive(THREAD_T id);
    void copy(THREAD_T id);
    gmacError_t migrate(Mode &mode, int acc);

    void addAccelerator(Accelerator *acc);

    static size_t totalMemory();
    size_t nAccelerators() const;
    bool allIntegrated();

    memory::ObjectMap &shared();
    const memory::ObjectMap &shared() const;
    memory::ObjectMap &replicated();
    const memory::ObjectMap &replicated() const;
    memory::ObjectMap &centralized();
    const memory::ObjectMap &centralized() const;
    memory::ObjectMap &orphans();
    const memory::ObjectMap &orphans() const;

    Mode *owner(const void *addr) const;
};

}

#include "Process.ipp"

#endif
