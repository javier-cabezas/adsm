#ifndef __MEMORY_OBJECT_IPP
#define __MEMORY_OBJECT_IPP

namespace gmac { namespace memory {

Object::Object(void *__addr, size_t __size) :
    __addr(__addr),
    __size(__size),
    __state(READONLY)
{ }

template<typename T>
SharedObject::SharedObject(size_t __size) :
    Object(NULL, __size)
{
}

}}

#endif
