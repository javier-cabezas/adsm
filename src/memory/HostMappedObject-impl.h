#ifndef GMAC_MEMORY_HOSTMAPPEDOBJECT_IMPL_H_
#define GMAC_MEMORY_HOSTMAPPEDOBJECT_IMPL_H_

namespace __impl { namespace memory {

inline HostMappedSet::HostMappedSet() :
    RWLock("HostMappedSet")
{}

inline HostMappedSet::~HostMappedSet()
{}


inline void *HostMappedObject::addr() const
{
    return addr_;
}

inline size_t HostMappedObject::size() const
{
    return size_;
}

inline void HostMappedObject::remove(void *addr)
{
    set_.remove(addr);
}

inline HostMappedObject *HostMappedObject::get(const void *addr)
{
    return set_.get((void *)addr);
}

}}

#endif
