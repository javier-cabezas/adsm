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
Manager::map(hostptr_t addr, size_t size, GmacProtection prot)
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

gmacError_t Manager::unmap(hostptr_t addr, size_t size)
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

gmacError_t Manager::alloc(hostptr_t *addr, size_t size)
{
    core::Mode &mode = core::Mode::current();
    // For integrated accelerators we want to use Centralized objects to avoid memory transfers
    if (mode.getAccelerator().integrated()) return hostMappedAlloc(addr, size);

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

gmacError_t Manager::hostMappedAlloc(hostptr_t *addr, size_t size)
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

gmacError_t Manager::globalAlloc(hostptr_t *addr, size_t size, GmacGlobalMallocType hint)
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
    
gmacError_t Manager::free(hostptr_t addr)
{
    gmacError_t ret = gmacSuccess;
    core::Mode &mode = core::Mode::current();
    Object *object = mode.getObject(addr);
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

accptr_t Manager::translate(const hostptr_t addr)
{
    __impl::core::Process &proc = __impl::core::Process::getInstance();
    accptr_t ret = proc.translate(addr);
    if(ret == NULL) {   
        HostMappedObject *object = HostMappedObject::get(addr);
        if(object != NULL) ret = object->acceleratorAddr(addr);
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
#ifdef USE_VM
    checkBitmapToAccelerator();
#endif
    return ret;
}


gmacError_t Manager::toIOBuffer(core::IOBuffer &buffer, const hostptr_t addr, size_t count)
{
    if (count > buffer.size()) return gmacErrorInvalidSize;
    core::Process &proc = core::Process::getInstance();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::Mode * mode = proc.owner(addr + off);
        if (mode == NULL) return gmacErrorInvalidValue;
        Object *obj = mode->getObject(addr + off);
        if (!obj) return gmacErrorInvalidValue;
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->addr();
        // Handle objects with no memory in the accelerator
		ret = obj->copyToBuffer(buffer, c, off, objOff);
		obj->release();
        if(ret != gmacSuccess) return ret;
        off += objCount;
        TRACE(LOCAL,"Copying from obj %p: "FMT_SIZE" of "FMT_SIZE, obj->addr(), c, count);
    } while(addr + off < addr + count);
    return ret;
}

gmacError_t Manager::fromIOBuffer(hostptr_t addr, core::IOBuffer &buffer, size_t count)
{
    if (count > buffer.size()) return gmacErrorInvalidSize;
    core::Process &proc = core::Process::getInstance();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::Mode *mode = proc.owner(addr + off);
        if (mode == NULL) return gmacErrorInvalidValue;
        Object *obj = mode->getObject(addr + off);
        if (!obj) return gmacErrorInvalidValue;
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->addr();
		ret = obj->copyFromBuffer(buffer, c, off, objOff);
		obj->release();        
        if(ret != gmacSuccess) return ret;
        off += objCount;
        TRACE(LOCAL,"Copying to obj %p: "FMT_SIZE" of "FMT_SIZE, obj->addr(), c, count);
    } while(addr + off < addr + count);
    return ret;
}

#ifdef USE_VM
void
Manager::checkBitmapToHost()
{
    core::Mode &mode = core::Mode::current();
    vm::SharedBitmap &acceleratorBitmap = mode.acceleratorDirtyBitmap();
    if (!acceleratorBitmap.synced()) {
        acceleratorBitmap.syncHost();
        mode.forEachObject(&Object::acquire);
    }
}

void
Manager::checkBitmapToAccelerator()
{
    core::Mode &mode = core::Mode::current();
    vm::SharedBitmap &acceleratorBitmap = mode.acceleratorDirtyBitmap();
    if (!acceleratorBitmap.clean()) {
        acceleratorBitmap.syncAccelerator();
    }
}
#endif

bool
Manager::read(hostptr_t addr)
{
#ifdef USE_VM
    checkBitmapToHost();
#endif
    core::Mode &mode = core::Mode::current();
    bool ret = true;
    Object *obj = mode.getObject(addr);
    if(obj == NULL) return false;
    TRACE(LOCAL,"Read access for object %p", obj->addr());
	gmacError_t err = obj->signalRead(addr);
    ASSERTION(err == gmacSuccess);
	obj->release();
    return ret;
}

bool
Manager::write(hostptr_t addr)
{
#ifdef USE_VM
    checkBitmapToHost();
#endif
    core::Mode &mode = core::Mode::current();
    bool ret = true;
    Object *obj = mode.getObject(addr);
    if(obj == NULL) return false;
    TRACE(LOCAL,"Write access for object %p", obj->addr());
	if(obj->signalWrite(addr) != gmacSuccess) ret = false;
	obj->release();
    return ret;
}

gmacError_t
Manager::memset(hostptr_t s, int c, size_t size)
{
    core::Process &proc = core::Process::getInstance();
    core::Mode *mode = proc.owner(s, size);
	if (mode == NULL) {
        ::memset(s, c, size);
        return gmacSuccess;
    }

    gmacError_t ret = gmacSuccess;

    Object *obj = mode->getObject(s);
    ASSERTION(obj != NULL);
    // Check for a fast path -- probably the user is just
    // initializing a single object or a portion of an object
    if(obj->addr() <= s && obj->end() >= (s + size)) {
        size_t objSize = (size < obj->size()) ? size : obj->size();
        ret = obj->memset(s - obj->addr(), c, objSize);
        obj->release();
        return ret;
    }

    // This code handles the case of the user initializing a portion of
    // memory that includes host memory and GMAC objects.
    size_t left = size;
    while(left > 0) {        
        // If there is no object, initialize the remaining host memory
        if(obj == NULL) {
            ::memset(s, c, left);
            left = 0; // This will finish the loop
        }
        else {
            // Check if there is a memory gap of host memory at the begining of the
            // memory range that remains to be initialized
            int gap = int(obj->addr() - s);
            if(gap > 0) { // If there is gap, initialize and advance the pointer
                ::memset(s, c, gap);
                left -= gap;
                s += gap;
                gap = 0;
            }
            // Check the size of the memory range from the current pointer to the end of the object
            // We add the gap, because if the ptr is within the object, its value will be negative
            size_t objSize = obj->size() + gap;
            // If the remaining memory in the object is larger than the remaining memory range, adjust
            // the size of the memory range to be initialized by the object
            objSize = (objSize < left) ? objSize : left;
            ret = obj->memset(s - obj->addr(), c, objSize);
            if(ret != gmacSuccess) break;
            left -= objSize; // Account for the bytes initialized by the object
            s += objSize;  // Advance the pointer
            obj->release();  // Release the object (it will not be needed anymore)
        }
        // Get the next object in the memory range that remains to be initialized
        obj = mode->getObject(s);
    }

    return ret;
}


gmacError_t
Manager::memcpyToObject(const Object &obj, const hostptr_t src, size_t size,
                        size_t objOffset)
{
    gmacError_t ret = gmacSuccess;

    core::Mode &mode = core::Mode::current();

    // We need to I/O buffers to double-buffer the copy
    core::IOBuffer *active = mode.createIOBuffer(obj.blockSize());
    core::IOBuffer *passive = mode.createIOBuffer(obj.blockSize());
    ASSERTION(active  != NULL);
    ASSERTION(passive != NULL);

    // Control variables
    size_t left = size;

    // Adjust the first copy to deal with a single block
    size_t copySize = size < obj.blockEnd(objOffset)? size: obj.blockEnd(objOffset);

    // Copy the data to the first block
    ::memcpy(active->addr(), src, copySize);

    hostptr_t ptr = src;
    while(left > 0) {
        // We do not need for the active buffer to be full because ::memcpy() is
        // a synchronous call
        ret = obj.copyFromBuffer(*active, copySize, 0, objOffset);
        ASSERTION(ret == gmacSuccess);
        //if (ret != gmacSuccess) return ret;
        ptr       += copySize;
        left      -= copySize;
        objOffset += copySize;
        if(left > 0) {
            // Start copying data from host memory to the passive I/O buffer
            copySize = (left < passive->size()) ? left : passive->size();
            passive->wait(); // Avoid overwritten a buffer that is already in use
            ::memcpy(passive->addr(), ptr, copySize);
        }
        // Swap buffers
        core::IOBuffer *tmp = active;
        active = passive;
        passive = tmp;
    }
    // Clean up buffers after they are idle
    passive->wait();
    mode.destroyIOBuffer(passive);
    active->wait();
    mode.destroyIOBuffer(active);

    return ret;
}

gmacError_t
Manager::memcpyToObject(const Object &dstObj, const Object &srcObj, size_t size,
                        size_t dstOffset, size_t srcOffset)
{
    gmacError_t ret = gmacSuccess;

    core::Mode &mode = core::Mode::current();

    // We need to I/O buffers to double-buffer the copy
    core::IOBuffer *active = mode.createIOBuffer(dstObj.blockSize());
    core::IOBuffer *passive = mode.createIOBuffer(dstObj.blockSize());
    ASSERTION(active  != NULL);
    ASSERTION(passive != NULL);

    // Control variables
    size_t left = size;

    // Adjust the first copy to deal with a single block
    size_t copySize = size < dstObj.blockEnd(dstOffset)? size: dstObj.blockEnd(dstOffset);

    // Single copy from the source to fill the buffer
    if (copySize <= srcObj.blockEnd(srcOffset)) {
        ret = srcObj.copyToBuffer(*active, copySize, 0, srcOffset);
        ASSERTION(ret == gmacSuccess);
        // if(ret != gmacSuccess) return ret;
    }
    else { // Two copies from the source to fill the buffer
        size_t copySize1 = srcObj.blockEnd(srcOffset);
        size_t copySize2 = copySize - copySize1;

        ret = srcObj.copyToBuffer(*active, copySize1, 0, srcOffset);
        ASSERTION(ret == gmacSuccess);
        //if(ret != gmacSuccess) return ret;
        ret = srcObj.copyToBuffer(*active, copySize2, copySize1, srcOffset + copySize1);
        ASSERTION(ret == gmacSuccess);
        //if(ret != gmacSuccess) return ret;
    }

    // Copy first chunk of data
    while(left > 0) {
        active->wait(); // Wait for the active buffer to be full
        ret = dstObj.copyFromBuffer(*active, copySize, 0, dstOffset);
        if(ret != gmacSuccess) return ret;
        left -= copySize;
        srcOffset += copySize;
        dstOffset += copySize;
        if(left > 0) {
            copySize = left < dstObj.blockSize()? left: dstObj.blockSize();
            // Avoid overwritting a buffer that is already in use
            passive->wait();

            // Request the next copy
            // Single copy from the source to fill the buffer
            if (copySize <= srcObj.blockEnd(srcOffset)) {
                ret = srcObj.copyToBuffer(*active, copySize, 0, srcOffset);
                ASSERTION(ret == gmacSuccess);
                //if(ret != gmacSuccess) return ret;
            }
            else { // Two copies from the source to fill the buffer
                size_t copySize1 = srcObj.blockEnd(srcOffset);
                size_t copySize2 = copySize - copySize1;

                ret = srcObj.copyToBuffer(*active, copySize1, 0, srcOffset);
                ASSERTION(ret == gmacSuccess);
                //if(ret != gmacSuccess) return ret;
                ret = srcObj.copyToBuffer(*active, copySize2, copySize1, srcOffset + copySize1);
                ASSERTION(ret == gmacSuccess);
                //if(ret != gmacSuccess) return ret;
            }
        }
        // Swap buffers
        core::IOBuffer *tmp = active;
        active = passive;
        passive = tmp;
    }
    // Clean up buffers after they are idle
    passive->wait();
    mode.destroyIOBuffer(passive);
    active->wait();
    mode.destroyIOBuffer(active);
    
    return ret;
}

gmacError_t
Manager::memcpyFromObject(hostptr_t dst, const Object &obj, size_t size,
                          size_t objOffset)
{
    gmacError_t ret = gmacSuccess;

    core::Mode &mode = core::Mode::current();
    // We need to I/O buffers to double-buffer the copy
    core::IOBuffer *active = mode.createIOBuffer(obj.blockSize());
    core::IOBuffer *passive = mode.createIOBuffer(obj.blockSize());
    ASSERTION(active  != NULL);
    ASSERTION(passive != NULL);

    // Control variables
    size_t left = size;

    // Adjust the first copy to deal with a single block
    size_t copySize = size < obj.blockEnd(objOffset)? size: obj.blockEnd(objOffset);

    // Copy the data to the first block
    ret = obj.copyToBuffer(*active, copySize, 0, objOffset);
    ASSERTION(ret == gmacSuccess);
    //if(ret != gmacSuccess) return ret;
    while(left > 0) {
        // Save values to use when copying the buffer to host memory
        size_t previousCopySize = copySize;
        left      -= copySize;
        objOffset += copySize;        
        if(left > 0) {
            // Start copying data from host memory to the passive I/O buffer
            copySize = (left < passive->size()) ? left : passive->size();
            // No need to wait for the buffer, because ::memcpy is a
            // synchronous call
            ret = obj.copyToBuffer(*passive, copySize, 0, objOffset);
            ASSERTION(ret == gmacSuccess);
            //if(ret != gmacSuccess) return ret;
        }        
        // Wait for the active buffer to be full
        active->wait();
        // Copy the active buffer to host
        ::memcpy(dst, active->addr(), previousCopySize);
        dst += previousCopySize;

        // Swap buffers
        core::IOBuffer *tmp = active;
        active = passive;
        passive = tmp;
    }
    // No need to wait for the buffers because we waited for them before ::memcpy
    mode.destroyIOBuffer(passive);
    mode.destroyIOBuffer(active);
    
    return ret;
}

size_t
Manager::hostMemory(hostptr_t addr, size_t size, const Object *obj) const
{
    // There is no object, so everything is in host memory
    if(obj == NULL) return size; 

    // The object starts after the memory range, return the difference
    if(addr < obj->addr()) return obj->addr() - addr;

    ASSERTION(obj->end() > addr); // Sanity check

    return 0;
}

gmacError_t
Manager::memcpy(hostptr_t dst, const hostptr_t src, size_t size)
{
    core::Process &proc = core::Process::getInstance();
    core::Mode *dstMode = proc.owner(dst, size);
    core::Mode *srcMode = proc.owner(src, size);

    if(dstMode == NULL && srcMode == NULL) {
        ::memcpy(dst, src, size);
        return gmacSuccess;
    }

    Object *dstObject = NULL;
    Object *srcObject = NULL;

    // Get initial objects
    if(dstMode != NULL) dstObject = dstMode->getObject(dst, size);
    if(srcMode != NULL) srcObject = srcMode->getObject(src, size);

    gmacError_t ret = gmacSuccess;
    size_t left = size;
    size_t offset = 0;
    size_t copySize = 0;
    while(left > 0) {
        // Get next objects involved, if necessary
        if(dstMode != NULL && dstObject != NULL && dstObject->end() < (dst + offset)) {
            dstObject->release();
            dstObject = dstMode->getObject(dst + offset, left);
        }
        if(srcMode != NULL && srcObject != NULL && srcObject->end() < (src + offset)) {
            srcObject->release();
            srcObject = srcMode->getObject(src + offset, left);
        }

        // Get the number of host-to-host memory we have to copy
        size_t dstHostMemory = hostMemory(dst + offset, left, dstObject);
        size_t srcHostMemory = hostMemory(src + offset, left, srcObject);

        if(dstHostMemory != 0 && srcHostMemory != 0) { // Host-to-host memory copy
            copySize = (dstHostMemory < srcHostMemory) ? dstHostMemory : srcHostMemory;
            ::memcpy(dst + offset, src + offset, copySize);
            ret = gmacSuccess;
        }
        else if(dstHostMemory != 0) { // Object-to-host memory copy
            size_t srcCopySize = srcObject->end() - src - offset;
            copySize = (dstHostMemory < srcCopySize) ? dstHostMemory : srcCopySize;
            size_t srcObjectOffset = src + offset - srcObject->addr();
            ret = memcpyFromObject(dst + offset, *srcObject, copySize, srcObjectOffset);
        }
        else if(srcHostMemory != 0) { // Host-to-object memory copy
            size_t dstCopySize = dstObject->end() - dst - offset;
            copySize = (srcHostMemory < dstCopySize) ? srcHostMemory : dstCopySize;
            size_t dstObjectOffset = dst + offset - dstObject->addr();
            ret = memcpyToObject(*dstObject, src + offset, copySize, dstObjectOffset);
        }
        else { // Object-to-object memory copy
            size_t srcCopySize = srcObject->end() - src - offset;
            size_t dstCopySize = dstObject->end() - dst - offset;
            copySize = (srcCopySize < dstCopySize) ? srcCopySize : dstCopySize;
            copySize = (copySize < left) ? copySize : left;
            size_t srcObjectOffset = src + offset - srcObject->addr();
            size_t dstObjectOffset = dst + offset - dstObject->addr();
            ret = memcpyToObject(*dstObject, *srcObject, copySize, dstObjectOffset, srcObjectOffset);
        }

        offset += copySize;
        left -= copySize;
    }
     
    if(dstObject != NULL) dstObject->release();
    if(srcObject != NULL) srcObject->release();
    
    return ret;
}

#if 0
gmacError_t Manager::moveTo(hostptr_t addr, core::Mode &mode)
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
