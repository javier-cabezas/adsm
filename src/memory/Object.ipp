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
    return addr_;
}

inline size_t
Object::size() const
{
    return size_;
}

inline void *
Object::start() const
{
    return addr_;
}

inline void *
Object::end() const
{
    return (void *)((uint8_t *)addr_ + size_);
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
