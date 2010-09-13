#include "Protocol.h"
#include "CentralizedObject.h"

namespace gmac { namespace memory {

#ifndef USE_MMAP
Object *Protocol::createCentralizedObject(size_t size)
{
    return new CentralizedObject(size);
}
#endif

}}

