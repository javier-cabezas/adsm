#ifndef __MEMORY_STATEOBJECT_IPP
#define __MEMORY_STATEOBJECT_IPP

namespace gmac { namespace memory {

template<typename T>
inline StateObject<T>::StateObject(size_t size) :
    Object(NULL, size)
{ }

template<typename T>
inline StateObject<T>::~StateObject()
{
    lockWrite();
    // Clean all system blocks
    typename SystemMap::const_iterator i;
    for(i = systemMap.begin(); i != systemMap.end(); i++)
        delete i->second;
    systemMap.clear();
    unlock();
}

template<typename T>
inline void StateObject<T>::setupSystem(T init)
{
    uint8_t *ptr = (uint8_t *)_addr;
    for(size_t i = 0; i < _size; i += paramPageSize, ptr += paramPageSize) {
        size_t blockSize = ((_size - i) > paramPageSize) ? paramPageSize : (_size - i);
        systemMap.insert(typename SystemMap::value_type(
            ptr + blockSize,
            new SystemBlock<T>(ptr, blockSize, init)));
    }
}

template<typename T>
inline SystemBlock<T> *StateObject<T>::findBlock(void *addr) 
{
    SystemBlock<T> *ret = NULL;
    lockRead();
    typename SystemMap::const_iterator block = systemMap.upper_bound(addr);
    if(block != systemMap.end()) ret = block->second;
    unlock();
    return ret;
}

template<typename T>
inline void StateObject<T>::state(T s)
{
    typename SystemMap::const_iterator i;
    lockWrite();
    for(i = systemMap.begin(); i != systemMap.end(); i++)
        i->second->state(s);
    unlock();
}

}}

#endif
