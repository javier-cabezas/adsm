#include "Protocol.h"
#include "CentralizedObject.h"

namespace gmac { namespace memory {

#ifndef USE_MMAP
Object *Protocol::createCentralizedObject(size_t size)
{
    Object *ret = new CentralizedObject(size);
    ret->init();
    return ret;
}
#endif

}}

