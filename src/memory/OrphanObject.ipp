#ifndef GMAC_MEMORY_ORPHANOBJECT_IPP
#define GMAC_MEMORY_ORPHANOBJECT_IPP

namespace gmac { namespace memory {

inline
OrphanObject::OrphanObject(const Object &obj) :
    Object(obj)
{
}

inline
OrphanObject::~OrphanObject()
{
}

inline
gmacError_t
OrphanObject::init()
{
    return gmacSuccess;
}

inline
void
OrphanObject::fini()
{
}

inline
void *
OrphanObject::getAcceleratorAddr(void *) const
{
    FATAL("Trying to get device address from orphan object");
    return NULL;
}

inline Mode &
OrphanObject::owner() const
{ 
    FATAL("Trying to get owner from orphan object");
    return gmac::Mode::current();
}

inline bool
OrphanObject::isLocal() const
{
    return false;
}

inline bool
OrphanObject::isInAccelerator() const
{
    return false;
}

}}

#endif
