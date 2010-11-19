#include "Protocol.h"
#include "CentralizedObject.h"

namespace __impl { namespace memory {

Protocol::~Protocol()
{
}

#ifndef USE_MMAP
Object *Protocol::createCentralizedObject(size_t size)
{
    Object *ret = new __impl::memory::CentralizedObject(size);
    ret->init();
    return ret;
}
#endif

}}

