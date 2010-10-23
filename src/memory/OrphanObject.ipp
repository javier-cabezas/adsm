#ifndef GMAC_MEMORY_ORPHANOBJECT_IPP
#define GMAC_MEMORY_ORPHANOBJECT_IPP

namespace gmac { namespace memory {

OrphanObject::OrphanObject(const Object &obj) :
    Object(obj)
{
}

OrphanObject::~OrphanObject()
{
}

}}

#endif
