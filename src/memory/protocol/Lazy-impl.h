#ifndef GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_

#include "memory/SharedObject.h"
#include "memory/DistributedObject.h"

namespace __impl { namespace memory { namespace protocol {

template<typename T>
inline Lazy<T>::Lazy(size_t limit) :
    gmac::memory::protocol::LazyBase(limit)
{}

template<typename T>
inline Lazy<T>::~Lazy()
{}

template<typename T>
inline memory::Object *Lazy<T>::createObject(core::Mode &current, size_t size, hostptr_t cpuPtr,
                                             GmacProtection prot, unsigned flags)
{
    Object *ret = new T(*this, current, cpuPtr,
                        size, LazyBase::state(prot));
    if(ret == NULL) return ret;
    if(ret->valid() == false) {
        ret->decRef();
        return NULL;
    }
    Memory::protect(ret->addr(), ret->size(), prot);
    LazyBase::limit_ += 2;
    return ret;
}

}}}
#endif
