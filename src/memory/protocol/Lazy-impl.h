#ifndef GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_

#include "memory/SharedObject.h"
#include "memory/DistributedObject.h"

namespace __impl { namespace memory { namespace protocol {

template<typename T>
inline Lazy<T>::Lazy(unsigned limit) :
    LazyBase(limit)
{}

template<typename T>
inline Lazy<T>::~Lazy()
{}

template<typename T>
inline memory::Object *Lazy<T>::createObject(size_t size, void *cpuPtr, 
                                             GmacProtection prot, unsigned flags)
{
    typename Object *ret = new T(*this, core::Mode::current(), cpuPtr, 
		size, state(prot));
	if(ret == NULL) return ret;
	if(ret->addr() == NULL) {
		ret->release();
		return NULL;
	}
	Memory::protect(ret->addr(), ret->size(), prot);
	return ret;
}

}}}
#endif