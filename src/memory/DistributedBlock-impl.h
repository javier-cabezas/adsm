#ifndef GMAC_MEMORY_DISTRIBUTEDBLOCK_INST_H_
#define GMAC_MEMORY_DISTRIBUTEDBLOCK_INST_H_

#include <algorithm>

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename State>
inline DistributedBlock<State>::DistributedBlock(Protocol &protocol, hostptr_t hostAddr,
											 hostptr_t shadowAddr, size_t size, typename State::ProtocolState init) :
    StateBlock<State>(protocol, hostAddr, shadowAddr, size, init)
{
}

template<typename State>
inline DistributedBlock<State>::~DistributedBlock()
{}

template<typename State>
inline void DistributedBlock<State>::addOwner(core::Mode &mode, accptr_t addr)
{
	StateBlock<State>::lock();

    AcceleratorMap::iterator it = acceleratorAddr_.find(addr);

    TRACE(LOCAL, "Adding owner for address for %u:%p @ Context %p", addr.pasId_, addr.get(), &mode);
    if (it == acceleratorAddr_.end()) {
        acceleratorAddr_.insert(AcceleratorMap::value_type(addr, std::list<core::Mode *>()));
        AcceleratorMap::iterator it = acceleratorAddr_.find(addr);
        it->second.push_back(&mode);

        if(StateBlock<State>::protocol_.needUpdate(*this) == true) {
#ifdef DEBUG
            gmacError_t ret =
#endif
                mode.copyToAccelerator(addr, StateBlock<State>::shadow_, StateBlock<State>::size_);
            ASSERTION(ret == gmacSuccess);
        }
    } else {
        it->second.push_back(&mode);
    }

    StateBlock<State>::unlock();
}

template<typename State>
inline void DistributedBlock<State>::removeOwner(core::Mode &mode)
{
	StateBlock<State>::lock();

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

	StateBlock<State>::unlock();
}

template<typename State>
inline core::Mode &DistributedBlock<State>::owner(core::Mode &current) const
{
    return current;
}

template<typename State>
inline accptr_t DistributedBlock<State>::acceleratorAddr(core::Mode &current, const hostptr_t addr) const
{
	accptr_t ret = accptr_t(0);

	StateBlock<State>::lock();
	AcceleratorMap::const_iterator i;
    TRACE(LOCAL, "Accelerator address for %p @ Context %p", addr, &current);
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        std::list<core::Mode *>::const_iterator it;
        it = std::find(list.begin(), list.end(), &current);

        if (it != list.end()) {
            ret = i->first + int(addr - StateBlock<State>::addr_);
            break;
        }

    }
    ASSERTION(i != acceleratorAddr_.end());
	StateBlock<State>::unlock();
	return ret;
}

template<typename State>
inline accptr_t DistributedBlock<State>::acceleratorAddr(core::Mode &current) const
{
	accptr_t ret = accptr_t(0);

	StateBlock<State>::lock();

    AcceleratorMap::const_iterator i;
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        if (std::find(list.begin(), list.end(), &current) != list.end()) {
            ret = i->first;
            break;
        }
    }

	StateBlock<State>::unlock();
	return ret;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::toHost(unsigned /*blockOff*/, size_t /*count*/)
{
	return gmacSuccess;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::toAccelerator(unsigned blockOff, size_t count)
{
    gmacError_t ret = gmacSuccess;
	AcceleratorMap::const_iterator i;
	for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        ASSERTION(list.size() > 0);
        core::Mode *mode = list.front();
		ret = mode->copyToAccelerator(i->first + blockOff, StateBlock<State>::shadow_ + blockOff, count);
		if(ret != gmacSuccess) break;
	}
	return ret;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::copyToHost(const hostptr_t src, size_t size, size_t blockOffset) const
{
    ::memcpy(StateBlock<State>::shadow_ + blockOffset, src, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::copyToHost(core::IOBuffer &buffer, size_t size, 
												size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(StateBlock<State>::shadow_ + blockOffset, buffer.addr() + bufferOffset, size);
	return gmacSuccess;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::copyToAccelerator(const hostptr_t src, size_t size,  size_t blockOffset) const
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

template<typename State>
inline gmacError_t DistributedBlock<State>::copyToAccelerator(core::IOBuffer &buffer, size_t size, 
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

template<typename State>
inline gmacError_t DistributedBlock<State>::copyFromHost(hostptr_t dst, size_t size, size_t blockOffset) const
{
    ::memcpy(dst, StateBlock<State>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::copyFromHost(core::IOBuffer &buffer, size_t size, 
												  size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(buffer.addr() + bufferOffset, StateBlock<State>::shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::copyFromAccelerator(hostptr_t dst, size_t size, size_t blockOffset) const
{
    ::memcpy(dst, StateBlock<State>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::copyFromAccelerator(core::IOBuffer &buffer, size_t size, 
													size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(buffer.addr() + bufferOffset, StateBlock<State>::shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::hostMemset(int v, size_t size, size_t blockOffset) const
{
    ::memset(StateBlock<State>::shadow_ + blockOffset, v, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::acceleratorMemset(int v, size_t size, size_t blockOffset) const
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
