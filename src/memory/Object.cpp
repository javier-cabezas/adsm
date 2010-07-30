#include "Object.h"

namespace gmac { namespace memory {

SharedObject::~SharedObject()
{
    lockWrite();
    if(__addr != NULL) {
        void *device = Map::current()->pageTable().translate(__addr);
        __owner->free(device);
        unmap(__addr);
        __addr = NULL;

        delete __accelerator;
        AcceleratorSet::const_iterator i;
        for(i = __system.begin(); i != __system.end(); i++)
            delete (*i);
        __system.clear();
    }
    unlock();
}


}}
