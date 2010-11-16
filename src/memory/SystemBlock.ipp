#ifndef GMAC_MEMORY_SYSTEMBLOCK_IPP_
#define GMAC_MEMORY_SYSTEMBLOCK_IPP_

namespace gmac { namespace memory { namespace __impl {

template<typename T>
inline SystemBlock<T>::SystemBlock(void *addr, size_t size, T state) :
    Block(addr, size),
    state_(state)
{
    TRACE(LOCAL,"Creating system block @ %p with %zd bytes", addr, size);
}

template<typename T>
inline SystemBlock<T>::~SystemBlock()
{ }


template<typename T>
inline T SystemBlock<T>::state() const
{ 
    T ret = state_;
    return ret;
}

template<typename T>
inline void SystemBlock<T>::state(T s)
{
    state_ = s;
}

}}}

#endif
