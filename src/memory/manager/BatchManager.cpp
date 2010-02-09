#include "BatchManager.h"

#include <debug.h>

#include "kernel/Context.h"

namespace gmac { namespace memory { namespace manager {

void BatchManager::release(void *addr)
{
    Region *reg = remove(ptr(addr));
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

void BatchManager::flush(const RegionVector & regions)
{
    RegionVector::const_iterator i;
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
    assert(0);
}

void
BatchManager::invalidate(const RegionVector & regions)
{
    assert(0);
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

Context *
BatchManager::owner(const void *)
{
    return NULL;
}

void
BatchManager::invalidate(const void *, size_t)
{
    assert(0);
}

void
BatchManager::flush(const void *, size_t)
{
    assert(0);
}

}}}
