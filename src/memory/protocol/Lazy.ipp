#ifndef __MEMORY_PROTOCOL_LAZY_IPP_
#define __MEMORY_PROTOCOL_LAZY_IPP_

namespace gmac { namespace memory { namespace protocol {

inline Lazy::Lazy()
{}

inline Lazy::~Lazy()
{ }

inline Object *Lazy::createObject(size_t size)
{
    return new SharedObject<Lazy::State>(size);
}

inline gmacError_t read(Object &o, void *addr)
{
    SharedObject<Lazy::State> &obj =
        dynamic_cast<SharedObject<Lazy::State> &>(o);

    return gmacSuccess;
}


} } }

#endif
