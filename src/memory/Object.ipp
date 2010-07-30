#ifndef __MEMORY_OBJECT_IPP
#define __MEMORY_OBJECT_IPP

#include <kernel/Mode.h>
#include <kernel/Context.h>

namespace gmac { namespace memory {


Object::Object(void *__addr, size_t __size) :
    RWLock(paraver::LockObject),
    __addr(__addr),
    __size(__size)
{ }

template<typename T>
SharedObject<T>::SharedObject(const Protocol &protocol, size_t size) :
    Object(NULL, __size),
    __owner(Mode::current())
{
    gmacError_t ret = gmacSuccess;
    void *device = NULL;
    // Allocate device and host memory
    ret = __owner->context().malloc(&device, size);
    __addr = map(device, size);
    if(__addr == NULL) __owner->context().free(device);
    // Create memory blocks
    __accelerator = new AcceleratorBlock(__owner, device, __size);
    uint8_t *ptr = (uint8_t *)__addr;
    for(size_t i = 0; i < __size; i += paramPageSize, ptr += paramPageSize) {
        __system.insert(protocol.createBlock(ptr, paramPageSize));
    }
}

}}

#endif
