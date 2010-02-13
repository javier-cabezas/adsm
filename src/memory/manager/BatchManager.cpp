#include "BatchManager.h"

#include <debug.h>

#include "kernel/Context.h"

namespace gmac { namespace memory { namespace manager {

void BatchManager::release(void *addr)
{
    Region *reg = remove(addr);
    hostUnmap(reg->start(), reg->size());
    removeVirtual(reg->start(), reg->size());
    delete reg;
}

void BatchManager::flush()
{
    Map::const_iterator i;
    Map * m = current();
    m->lock();
    for(i = m->begin(); i != m->end(); i++) {
        TRACE("Memory Copy to Device");
        Context::current()->copyToDevice(ptr(i->second->start()),
                                         i->second->start(),
                                         i->second->size());
    }
    m->unlock();
    /*!
      \todo Fix vm
    */
    //Context::current()->flush();
    Context::current()->sync();
}

void BatchManager::flush(const RegionSet & regions)
{
    if (regions.size() == 0) {
        flush();
        return;
    }

    RegionSet::const_iterator i;
    Map * m = current();
    m->lock();
    for(i = regions.begin(); i != regions.end(); i++) {
        TRACE("Memory Copy to Device");
        Context::current()->copyToDevice(ptr((*i)->start()),
                                         (*i)->start(),
                                         (*i)->size());
    }
    m->unlock();
    /*!
      \todo Fix vm
    */
    //Context::current()->flush();
    Context::current()->sync();
}

void
BatchManager::invalidate()
{
    // Do nothing
}

void
BatchManager::invalidate(const RegionSet & regions)
{
    // Do nothing
}

void BatchManager::sync()
{
    Map::const_iterator i;
    current()->lock();
    for(i = current()->begin(); i != current()->end(); i++) {
        TRACE("Memory Copy from Device");
        Context::current()->copyToHost(i->second->start(),
                                       ptr(i->second->start()), i->second->size());
    }
    current()->unlock();
}

void
BatchManager::invalidate(const void *, size_t)
{
    // Do nothing
}

void
BatchManager::flush(const void *, size_t)
{
    // Do nothing
}

}}}
