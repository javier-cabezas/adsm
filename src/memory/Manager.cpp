#include "core/IOBuffer.h"
#include "core/Process.h"

#include "Manager.h"
#include "Map.h"
#include "Object.h"
#include "HostMappedObject.h"

namespace __impl { namespace memory {

Manager::Manager()
{
    TRACE(LOCAL,"Memory manager starts");
}

Manager::~Manager()
{
    TRACE(LOCAL,"Memory manager finishes");
}

#if 0
gmacError_t
Manager::map(void *addr, size_t size, GmacProtection prot)
{
    core::Mode &mode = core::Mode::current();

    // Create new shared object
    Object *object = protocol_->createSharedObject(size, addr, prot);
    if(object == NULL) {
        return gmacErrorMemoryAllocation;
    }

    // Insert object into memory maps
    mode.addObject(*object);

    return gmacSuccess;
}

gmacError_t Manager::unmap(void *addr, size_t size)
{
    // TODO implement partial unmapping
    gmacError_t ret = gmacSuccess;
    core::Mode &mode = core::Mode::current();
    Object *object = mode.getObjectWrite(addr);
    if(object != NULL)  {
        if (object->isInAccelerator()) {
            ret = protocol_->toHost(*object);
            if (ret != gmacSuccess) { 
                mode.putObject(*object);
                return ret;
            }
            protocol_->deleteObject(*object);
            if (ret != gmacSuccess) { 
                mode.putObject(*object);
                return ret;
            }
        }
        // TODO capture all the possible errors
        mode.removeObject(*object);
        mode.putObject(*object);
        object->fini();
        delete object;
    }
    else ret = gmacErrorInvalidValue;
    return ret;
}
#endif
gmacError_t Manager::alloc(void **addr, size_t size)
{
    core::Mode &mode = core::Mode::current();
    // For integrated devices we want to use Centralized objects to avoid memory transfers
    //if (mode.integrated()) return globalAlloc(addr, size, GMAC_GLOBAL_MALLOC_CENTRALIZED);

    // Create new shared object
    Object *object = mode.protocol().createObject(size, NULL, GMAC_PROT_READ, 0);
    *addr = object->addr();
    if(*addr == NULL) {
		object->release();
        return gmacErrorMemoryAllocation;
    }

    // Insert object into memory maps
    mode.addObject(*object);
    object->release();
    return gmacSuccess;
}

gmacError_t Manager::hostMappedAlloc(void **addr, size_t size)
{
    gmacError_t ret = gmacSuccess;
    HostMappedObject *object = new HostMappedObject(size);
    *addr = object->addr();
    if(*addr == NULL) {
        delete object;
        return gmacErrorMemoryAllocation;
    }
    return gmacSuccess;
}

gmacError_t Manager::globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint)
{
    core::Process &proc = core::Process::getInstance();
    core::Mode &mode = core::Mode::current();

    // If a centralized object is requested, try creating it
    if(hint == GMAC_GLOBAL_MALLOC_CENTRALIZED) {
        gmacError_t ret = hostMappedAlloc(addr, size);
        if(ret == gmacSuccess) return ret;
    }

    Object *object = proc.protocol().createObject(size, NULL, GMAC_PROT_READ, 0);
    *addr = object->addr();
    if(*addr == NULL) {
        object->release();
        return hostMappedAlloc(addr, size); // Try using a host mapped object
    }
    gmacError_t ret = proc.globalMalloc(*object, size);
    object->release();
    return ret;
}
    
gmacError_t Manager::free(void * addr)
{
    gmacError_t ret = gmacSuccess;
    core::Mode &mode = core::Mode::current();
    const Object *object = mode.getObject(addr);
    if(object != NULL)  {
        mode.removeObject(*object);
		object->release();
    }
    else {
        HostMappedObject *hostMappedObject = HostMappedObject::get(addr);
        if(hostMappedObject == NULL) return gmacErrorInvalidValue;
        hostMappedObject->release();
    }
    return ret;
}

void *Manager::translate(const void *addr)
{
    __impl::core::Process &proc = __impl::core::Process::getInstance();
    void *ret = proc.translate(addr);
    if(ret == NULL) {   
        HostMappedObject *object = HostMappedObject::get(addr);
        if(object != NULL) ret = object->deviceAddr(addr);
    }
    return ret;
}

gmacError_t Manager::acquireObjects()
{
    gmacError_t ret = gmacSuccess;
    core::Mode &mode = core::Mode::current();
    if(mode.releasedObjects() == false) {
        return gmacSuccess;
    }
	mode.forEachObject(&Object::acquire);
    mode.acquireObjects();
    return ret;
}

gmacError_t Manager::releaseObjects()
{
#ifdef USE_VM
    checkBitmapToDevice();
#endif
    core::Mode &mode = core::Mode::current();
    TRACE(LOCAL,"Releasing Objects");
    gmacError_t ret = gmacSuccess;
    // Release per-mode objects
    ret = mode.protocol().releaseObjects();
    if(ret == gmacSuccess) {
        // Release global per-process objects
        core::Process::getInstance().protocol().releaseObjects();
        mode.releaseObjects();
    }
    return ret;
}


gmacError_t Manager::toIOBuffer(core::IOBuffer &buffer, const void *addr, size_t count)
{
    if (count > buffer.size()) return gmacErrorInvalidSize;
    core::Process &proc = core::Process::getInstance();
    gmacError_t ret = gmacSuccess;
    const uint8_t *ptr = (const uint8_t *)addr;
    unsigned off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::Mode * mode = proc.owner(ptr + off);
        if (mode == NULL) return gmacErrorInvalidValue;
        const Object *obj = mode->getObject(ptr + off);
        if (!obj) return gmacErrorInvalidValue;
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (ptr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        unsigned objOff = unsigned(ptr - obj->addr());
        // Handle objects with no memory in the accelerator
		ret = obj->copyToBuffer(buffer, c, off, objOff);
		obj->release();
        if(ret != gmacSuccess) return ret;
        off += unsigned(objCount);
        TRACE(LOCAL,"Copying from obj %p: "FMT_SIZE" of "FMT_SIZE, obj->addr(), c, count);
    } while(ptr + off < ptr + count);
    return ret;
}

gmacError_t Manager::fromIOBuffer(void * addr, core::IOBuffer &buffer, size_t count)
{
    if (count > buffer.size()) return gmacErrorInvalidSize;
    core::Process &proc = core::Process::getInstance();
    gmacError_t ret = gmacSuccess;
    uint8_t *ptr = (uint8_t *)addr;
    unsigned off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::Mode *mode = proc.owner(ptr + off);
        if (mode == NULL) return gmacErrorInvalidValue;
        const Object *obj = mode->getObject(ptr + off);
        if (!obj) return gmacErrorInvalidValue;
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (ptr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        unsigned objOff = unsigned(ptr - obj->addr());
		ret = obj->copyFromBuffer(buffer, c, off, objOff);
		obj->release();        
        if(ret != gmacSuccess) return ret;
        off += unsigned(objCount);
        TRACE(LOCAL,"Copying to obj %p: "FMT_SIZE" of "FMT_SIZE, obj->addr(), c, count);
    } while(ptr + off < ptr + count);
    return ret;
}

#ifdef USE_VM
void Manager::checkBitmapToHost()
{
    core::Mode &mode = core::Mode::current();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    if (!bitmap.synced()) {
        bitmap.syncHost();
        mode.forEachObject(&Object::acquire);
    }
}

void Manager::checkBitmapToDevice()
{
    core::Mode &mode = core::Mode::current();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    if (!bitmap.clean()) {
        bitmap.syncDevice();
    }
}
#endif

bool Manager::read(void *addr)
{
    core::Mode &mode = core::Mode::current();
#ifdef USE_VM
    checkBitmapToHost();
#endif
    bool ret = true;
    const Object *obj = mode.getObject(addr);
    if(obj == NULL) return false;
    TRACE(LOCAL,"Read access for object %p", obj->addr());
	gmacError_t err = obj->signalRead(addr);
    ASSERTION(err == gmacSuccess);
	obj->release();
    return ret;
}

bool Manager::write(void *addr)
{
    core::Mode &mode = core::Mode::current();
#ifdef USE_VM
    checkBitmapToHost();
#endif
    bool ret = true;
    const Object *obj = mode.getObject(addr);
    if(obj == NULL) return false;
    TRACE(LOCAL,"Write access for object %p", obj->addr());
	if(obj->signalWrite(addr) != gmacSuccess) ret = false;
	obj->release();
    return ret;
}

gmacError_t Manager::memset(void *s, int c, size_t n)
{
    core::Process &proc = core::Process::getInstance();
    core::Mode *mode = proc.owner(s);
	if (mode == NULL) {
        ::memset(s, c, n);
        return gmacSuccess;
    }
    // TODO: deal with the case of several objects being affected
    const Object * obj = mode->getObject(s);
    ASSERTION(obj != NULL);
    gmacError_t ret = obj->memset(s, c, n);    
    obj->release();
    return ret;
}


gmacError_t Manager::memcpy(void *dst, const void *src, size_t n)
{
    core::Process &proc = core::Process::getInstance();
    core::Mode *dstMode = proc.owner(dst);
    core::Mode *srcMode = proc.owner(src);

    if(dstMode == NULL && srcMode == NULL) {
        ::memcpy(dst, src, n);
        return gmacSuccess;
    }
    // TODO: consider the case of a memcpy not starting on an object
    const Object *dstObject = NULL;
    const Object *srcObject = NULL;
	if(dstMode != NULL) {
        dstObject = dstMode->getObject(dst);
        ASSERTION(dstObject != NULL);
    }

    if(srcMode != NULL) {
        srcObject = srcMode->getObject(src);
        ASSERTION(srcObject != NULL);
    }

    gmacError_t ret = gmacSuccess;
    if(dstObject != NULL) {
        if(srcObject == NULL) ret = dstObject->memcpyFromMemory(dst, src, n);
        else {
            unsigned objectOffset = unsigned((uint8_t *)srcObject->addr() - (uint8_t *)src);
            ret = dstObject->memcpyFromObject(dst, *srcObject, n, objectOffset);
        }
    }
    else ret = srcObject->memcpyToMemory(dst, src, n);

    if(dstObject != NULL) dstObject->release();
    if(srcObject != NULL) srcObject->release();
    
    return ret;
}

#if 0
gmacError_t
Manager::removeMode(core::Mode &mode)
{
    gmacError_t ret = protocol_->removeMode(mode);
    return ret;
}

gmacError_t Manager::moveTo(void * addr, core::Mode &mode)
{
    Object * obj = mode.getObjectWrite(addr);
    if(obj == NULL) return gmacErrorInvalidValue;

    mode.putObject(*obj);
#if 0
    StateObject<T>::lockWrite();
    typename StateObject<T>::SystemMap::iterator i;
    int idx = 0;
    for(i = StateObject<T>::systemMap.begin(); i != StateObject<T>::systemMap.end(); i++) {
        gmacError_t ret = accelerator->get(idx++ * paramPageSize, i->second);
    }

    owner_->free(accelerator->addr());
    
    StateObject<T>::unlock();
#endif
    return gmacSuccess;
}
#endif
}}
