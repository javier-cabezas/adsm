#ifndef GMAC_MEMORY_DISTRIBUTEDBLOCK_IMPL_H_
#define GMAC_MEMORY_DISTRIBUTEDBLOCK_IMPL_H_

#include <algorithm>

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename T>
inline DistributedBlock<T>::DistributedBlock(Protocol &protocol, hostptr_t hostAddr,
											 hostptr_t shadowAddr, size_t size, T init) :
    StateBlock<T>(protocol, hostAddr, shadowAddr, size, init)
{
}

template<typename T>
inline DistributedBlock<T>::~DistributedBlock()
{}

template<typename T>
inline void DistributedBlock<T>::addOwner(core::Mode &mode, accptr_t addr)
{
	StateBlock<T>::lock();

    AcceleratorMap::iterator it = acceleratorAddr_.find(addr);

    TRACE(LOCAL, "Adding owner for address for %u:%p @ Context %p", addr.pasId_, addr.get(), &mode);
    if (it == acceleratorAddr_.end()) {
        acceleratorAddr_.insert(AcceleratorMap::value_type(addr, std::list<core::Mode *>()));
        AcceleratorMap::iterator it = acceleratorAddr_.find(addr);
        it->second.push_back(&mode);

        if(StateBlock<T>::protocol_.needUpdate(*this) == true) {
            gmacError_t ret = mode.copyToAccelerator(addr, StateBlock<T>::shadow_, StateBlock<T>::size_);
            ASSERTION(ret == gmacSuccess);
        }
    } else {
        it->second.push_back(&mode);
    }

    StateBlock<T>::unlock();
}

template<typename T>
inline void DistributedBlock<T>::removeOwner(core::Mode &mode)
{
	StateBlock<T>::lock();

    AcceleratorMap::iterator i;
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        std::list<core::Mode *> &list = i->second;
        std::list<core::Mode *>::iterator j = std::find(list.begin(), list.end(), &mode);
        if (j != list.end()) {
            list.erase(j);
            if (list.size() == 0) acceleratorAddr_.erase(i);
            break;
        }
    }

	StateBlock<T>::unlock();
}

template<typename T>
inline core::Mode &DistributedBlock<T>::owner() const
{
	return core::Mode::getCurrent();
}

template<typename T>
inline accptr_t DistributedBlock<T>::acceleratorAddr(const hostptr_t addr) const
{
	accptr_t ret = accptr_t(NULL);

	StateBlock<T>::lock();
	AcceleratorMap::const_iterator i;
    core::Mode &mode = core::Mode::getCurrent();
    TRACE(LOCAL, "Accelerator address for %p @ Context %p", addr, &mode);
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        std::list<core::Mode *>::const_iterator it;
        it = std::find(list.begin(), list.end(), &mode);

        if (it != list.end()) {
            ret = i->first + int(addr - StateBlock<T>::addr_);
            break;
        }

    }
    ASSERTION(i != acceleratorAddr_.end());
	StateBlock<T>::unlock();
	return ret;
}

template<typename T>
inline accptr_t DistributedBlock<T>::acceleratorAddr() const
{
	accptr_t ret = accptr_t(NULL);

	StateBlock<T>::lock();

    AcceleratorMap::const_iterator i;
    core::Mode &mode = core::Mode::getCurrent();
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        if (std::find(list.begin(), list.end(), &mode) != list.end()) {
            ret = i->first;
            break;
        }
    }

	StateBlock<T>::unlock();
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::toHost() const
{
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::toAccelerator()
{
	gmacError_t ret = gmacSuccess;
	AcceleratorMap::const_iterator i;
	for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        ASSERTION(list.size() > 0);
        core::Mode *mode = list.front();
		ret = mode->copyToAccelerator(i->first, StateBlock<T>::shadow_, StateBlock<T>::size_);
		if(ret != gmacSuccess) break;
	}
#ifdef USE_VM
    Block::resetBitmapStats();
#endif
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToHost(const hostptr_t src, size_t size, size_t blockOffset) const
{
    ::memcpy(StateBlock<T>::shadow_ + blockOffset, src, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToHost(core::IOBuffer &buffer, size_t size, 
												size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(StateBlock<T>::shadow_ + blockOffset, buffer.addr() + bufferOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToAccelerator(const hostptr_t src, size_t size,  size_t blockOffset) const
{
    gmacError_t ret = gmacSuccess;
	AcceleratorMap::const_iterator i;
	for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        ASSERTION(list.size() > 0);
        core::Mode *mode = list.front();
		ret = mode->copyToAccelerator(i->first + ptroff_t(blockOffset), src, size);
		if(ret != gmacSuccess) return ret;
	}
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToAccelerator(core::IOBuffer &buffer, size_t size, 
												  size_t bufferOffset, size_t blockOffset) const
{
	gmacError_t ret = gmacSuccess;
	AcceleratorMap::const_iterator i;
	for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        ASSERTION(list.size() > 0);
        core::Mode *mode = list.front();
		ret = mode->bufferToAccelerator(i->first + ptroff_t(blockOffset), buffer, size, bufferOffset);
		if(ret != gmacSuccess) return ret;
	}
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromHost(hostptr_t dst, size_t size, size_t blockOffset) const
{
    ::memcpy(dst, StateBlock<T>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromHost(core::IOBuffer &buffer, size_t size, 
												  size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(buffer.addr() + bufferOffset, StateBlock<T>::shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromAccelerator(hostptr_t dst, size_t size, size_t blockOffset) const
{
    ::memcpy(dst, StateBlock<T>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromAccelerator(core::IOBuffer &buffer, size_t size, 
													size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(buffer.addr() + bufferOffset, StateBlock<T>::shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::hostMemset(int v, size_t size, size_t blockOffset) const
{
    ::memset(StateBlock<T>::shadow_ + blockOffset, v, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::acceleratorMemset(int v, size_t size, size_t blockOffset) const
{
    gmacError_t ret = gmacSuccess;
	AcceleratorMap::const_iterator i;
	for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        ASSERTION(list.size() > 0);
        core::Mode *mode = list.front();
        ret = mode->memset(i->first + ptroff_t(blockOffset), v, size);
		if(ret != gmacSuccess) break;
	}
	return ret;
}

}}

#endif
