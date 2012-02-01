#ifndef GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_

#include "memory/object_state.h"

namespace __impl { namespace memory { namespace protocols {

template<typename T>
inline lazy<T>::lazy(bool eager) :
    gmac::memory::protocols::lazy_base(eager)
{}

template<typename T>
inline lazy<T>::~lazy()
{}

template<typename T>
memory::object *
lazy<T>::create_object(host_ptr cpuPtr, size_t size,
                       int flagsHost, int flagsDevice, gmacError_t &err)
{
    object *ret = new T(*this, cpuPtr, size,
                        flagsHost, flagsDevice,
                        lazy_base::state(flagsHost, flagsDevice), err);
    if (ret == NULL) return ret;
    if (err != gmacSuccess) {
        return NULL;
    }
    //memory_ops::protect(ret->get_bounds().start, ret->size(), prot);

    return ret;
}

}}}

#endif
