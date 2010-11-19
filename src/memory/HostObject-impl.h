#ifndef GMAC_MEMORY_HOSTOBJECT_IMPL_H_
#define GMAC_MEMORY_HOSTOBJECT_IMPL_H_

namespace __impl { namespace memory {

inline HostObject::HostObject(size_t size) :
	Object(NULL, size)
{ }

inline HostObject::~HostObject()
{}

inline Protocol &HostObject::protocol()
{
	return protocol_;
}

}}

#endif
