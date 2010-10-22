#ifndef GMAC_MEMORY_OBJECT_IPP_
#define GMAC_MEMORY_OBJECT_IPP_

namespace gmac { namespace memory {

inline
Object::Object(void *addr, size_t size) :
    util::RWLock("memory::Object"), addr_(addr), size_(size)
{
}

inline
Object::~Object()
{
}

inline uint8_t *
Object::addr() const
{
    return (uint8_t *) addr_;
}

inline uint8_t *
Object::end() const
{
    return addr() + size_;
}

inline size_t
Object::size() const
{
    return size_;
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
