#ifndef GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_

#include "memory/object_state.h"

namespace __impl { namespace memory { namespace protocol {

template<typename T>
inline Lazy<T>::Lazy(bool eager) :
    gmac::memory::protocol::LazyBase(eager)
{}

template<typename T>
inline Lazy<T>::~Lazy()
{}

template<typename T>
memory::object *
Lazy<T>::createObject(size_t size, hostptr_t cpuPtr,
                      GmacProtection prot, unsigned flags)
{
    gmacError_t err;
    object *ret = new T(*this, cpuPtr, size, LazyBase::state(prot), err);
    if(ret == NULL) return ret;
    if(err != gmacSuccess) {
        ret->decRef();
        return NULL;
    }
    Memory::protect(ret->addr(), ret->size(), prot);

    return ret;
}

}}}

#endif
