#ifndef GMAC_MEMORY_STATEOBJECT_IPP
#define GMAC_MEMORY_STATEOBJECT_IPP

namespace gmac { namespace memory {

template<typename T>
inline StateObject<T>::StateObject(size_t size, T init) :
    Object(NULL, size), init_(init)
{ }

template<typename T>
inline StateObject<T>::~StateObject()
{
    // Clean all system blocks
    typename SystemMap::const_iterator i;
    for(i = systemMap.begin(); i != systemMap.end(); i++)
        delete i->second;
    systemMap.clear();
}

template<typename T>
inline void StateObject<T>::setupSystem()
{
    uint8_t *ptr = (uint8_t *)addr_;
    for(size_t i = 0; i < size_; i += paramPageSize, ptr += paramPageSize) {
        size_t blockSize = ((size_ - i) > paramPageSize) ? paramPageSize : (size_ - i);
        systemMap.insert(typename SystemMap::value_type(
            ptr + blockSize,
            new SystemBlock<T>(ptr, blockSize, init_)));
    }
}

template<typename T>
inline SystemBlock<T> *StateObject<T>::findBlock(void *addr) const
{
    SystemBlock<T> *ret = NULL;
    typename SystemMap::const_iterator block = systemMap.upper_bound(addr);
    if(block != systemMap.end()) ret = block->second;
    return ret;
}

template<typename T>
inline void StateObject<T>::state(T s)
{
    trace("Setting state of %p to %d", addr_, s);
    typename SystemMap::const_iterator i;
    for(i = systemMap.begin(); i != systemMap.end(); i++)
        i->second->state(s);
}

}}

#endif
