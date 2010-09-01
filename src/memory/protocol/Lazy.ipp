#ifndef __MEMORY_PROTOCOL_LAZY_IPP_
#define __MEMORY_PROTOCOL_LAZY_IPP_

namespace gmac { namespace memory { namespace protocol {

inline Lazy::Lazy()
{}

inline Lazy::~Lazy()
{ }

inline Object *Lazy::createObject(size_t size)
{
    return new SharedObject<Lazy::State>(size, ReadOnly);
}

inline Object *Lazy::createReplicatedObject(size_t size)
{
    return new ReplicatedObject<Lazy::State>(size, ReadOnly);
}

} } }

#endif
