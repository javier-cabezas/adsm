#ifndef __MEMORY_OBJECT_IPP_
#define __MEMORY_OBJECT_IPP_

namespace gmac { namespace memory {

inline
Object::~Object()
{
}

inline void *
Object::addr() const
{
    return _addr;
}

inline size_t
Object::size() const
{
    return _size;
}

inline void *
Object::start() const
{
    return _addr;
}

inline void *
Object::end() const
{
    return (void *)((uint8_t *)_addr + _size);
}

inline gmacError_t
Object::free()
{
    return gmacSuccess;
}

inline gmacError_t
Object::realloc(Mode &mode)
{
    return gmacSuccess;
}

}}

#endif
