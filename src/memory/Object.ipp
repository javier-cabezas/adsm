#ifndef __MEMORY_OBJECT_IPP
#define __MEMORY_OBJECT_IPP

namespace gmac { namespace memory {


Object::Object(void *__addr, size_t __size) :
    RWLock(paraver::LockObject),
    __addr(__addr),
    __size(__size)
{ }

template<typename T>
SharedObject::SharedObject(const Protocol &protocol, size_t __size) :
    Object(NULL, __size),
    __owner(Context::current())
{
    gmacError_t ret = gmacSuccess;
    void *device = NULL;
    // Allocate device and host memory
    ret = __owner->malloc(&device, size);
    __addr = map(device, size);
    if(__addr == NULL) __owner->free(device);
    // Insert translation into the thread's page table
    Map::current()->pageTable().insert(__addr, deviceAddr);
    // Create memory blocks
    __accelerator = new AcceleratorBlock(device, __size);
    uint8_t *ptr = (uint8_t *)__addr;
    for(size_t i = 0; i < __size; i += BlockSize, ptr += BlockSize) {
        __system.insert(protocol.createBlock(ptr, BlockSize));
    }
}

}}

#endif
