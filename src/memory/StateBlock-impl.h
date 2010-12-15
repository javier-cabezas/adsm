#ifndef GMAC_MEMORY_STATEBLOCK_IMPL_H_
#define GMAC_MEMORY_STATEBLOCK_IMPL_H_

namespace __impl { namespace memory { 

template<typename T>
inline StateBlock<T>::StateBlock(Protocol &protocol, hostptr_t addr, 
                                 hostptr_t shadow, size_t size, T init) :
	gmac::memory::Block(protocol, addr, shadow, size),
	state_(init)
{}

template<typename T>
inline const T &StateBlock<T>::state() const
{
	return state_;
}

template<typename T>
inline void StateBlock<T>::state(const T &s)
{
	state_ = s;
}


}}

#endif
