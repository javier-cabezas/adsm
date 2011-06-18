#include "core/IOBuffer.h"
#include "core/Mode.h"
#include "core/Process.h"

#include "memory/Handler.h"
#include "memory/HostMappedObject.h"
#include "memory/Manager.h"
#include "memory/Object.h"

using __impl::util::params::ParamAutoSync;

namespace __impl { namespace memory {

Manager::Manager(core::Process &proc) :
    proc_(proc)
{
    TRACE(LOCAL,"Memory manager starts");
    Init();
    Handler::setManager(*this);
}

Manager::~Manager()
{
}


gmacError_t Manager::alloc(core::Mode &mode, hostptr_t *addr, size_t size)
{
    trace::EnterCurrentFunction();
    // For integrated accelerators we want to use Centralized objects to avoid memory transfers
    // TODO: ask process instead
    // if (mode.getAccelerator().integrated()) return hostMappedAlloc(addr, size);

    // Create new shared object. We set the memory as invalid to avoid stupid data transfers
    // to non-initialized objects
    Object *object = mode.protocol().createObject(mode, size, NULL, GMAC_PROT_READ, 0);
    if(object == NULL) {
        trace::ExitCurrentFunction();
        return gmacErrorMemoryAllocation;
    }
    *addr = object->addr();

    // Insert object into memory maps
    mode.addObject(*object);
    object->release();
    trace::ExitCurrentFunction();
    return gmacSuccess;
}

gmacError_t Manager::hostMappedAlloc(core::Mode &mode, hostptr_t *addr, size_t size)
{
    trace::EnterCurrentFunction();
    HostMappedObject *object = new HostMappedObject(mode, size);
    *addr = object->addr();
    if(*addr == NULL) {
        delete object;
        trace::ExitCurrentFunction();
        return gmacErrorMemoryAllocation;
    }
    trace::ExitCurrentFunction();
    return gmacSuccess;
}

gmacError_t Manager::globalAlloc(core::Mode &mode, hostptr_t *addr, size_t size, GmacGlobalMallocType hint)
{
    trace::EnterCurrentFunction();

    // If a centralized object is requested, try creating it
    if(hint == GMAC_GLOBAL_MALLOC_CENTRALIZED) {
        gmacError_t ret = hostMappedAlloc(mode, addr, size);
        if(ret == gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
    }
    Protocol *protocol = proc_.protocol();
    if(protocol == NULL) return gmacErrorInvalidValue;
    Object *object = protocol->createObject(mode, size, NULL, GMAC_PROT_NONE, 0);
    *addr = object->addr();
    if(*addr == NULL) {
        object->release();
        trace::ExitCurrentFunction();
        return hostMappedAlloc(mode, addr, size); // Try using a host mapped object
    }
    gmacError_t ret = proc_.globalMalloc(*object);
    object->release();
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::free(core::Mode &mode, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    Object *object = mode.getObject(addr);
    if(object != NULL)  {
        mode.removeObject(*object);
        object->release();
    }
    else {
        HostMappedObject *hostMappedObject = HostMappedObject::get(addr);
        if(hostMappedObject == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        hostMappedObject->release();
        // We need to release the object twice to effectively destroy it
        HostMappedObject::remove(addr);
        hostMappedObject->release();
    }
    trace::ExitCurrentFunction();
    return ret;
}

accptr_t Manager::translate(core::Mode &mode, const hostptr_t addr)
{
    trace::EnterCurrentFunction();
    accptr_t ret = proc_.translate(addr);
    if(ret == 0) {
        HostMappedObject *object = HostMappedObject::get(addr);
        if(object != NULL) {
            ret = object->acceleratorAddr(addr);
            object->release();
        }
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::acquireObjects(core::Mode &mode)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    if(mode.validObjects() && mode.releasedObjects()) {
        TRACE(LOCAL,"Acquiring Objects");
        mode.forEachObject(&Object::acquire);
        mode.acquireObjects();
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::releaseObjects(core::Mode &mode)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    if(mode.validObjects()) {
        TRACE(LOCAL,"Releasing Objects");
        // Release per-mode objects
        ret = mode.protocol().releaseObjects();
        mode.releaseObjects();
        if(ret == gmacSuccess) {
            // Release global per-process objects
            Protocol *protocol = proc_.protocol();
            if(protocol != NULL) protocol->releaseObjects();
        }
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::releaseObjects(core::Mode &mode, const std::list<hostptr_t> &objects)
{
    // TODO: release the given objects only
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    if(mode.validObjects()) {
        TRACE(LOCAL,"Releasing Objects");
        // Release per-mode objects
        ret = mode.protocol().releaseObjects();
        mode.releaseObjects();
        if(ret == gmacSuccess) {
            // Release global per-process objects
            Protocol *protocol = proc_.protocol();
            if(protocol != NULL) protocol->releaseObjects();
        }
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::toIOBuffer(core::Mode &mode, core::IOBuffer &buffer, size_t bufferOff, const hostptr_t addr, size_t count)
{
    if (count > (buffer.size() - bufferOff)) return gmacErrorInvalidSize;
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::Mode * mode = proc_.owner(addr + off);
        if (mode == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }

#ifdef USE_VM
        CFATAL(mode->releasedObjects() == false, "Acquiring bitmap on released objects");
        vm::Bitmap &bitmap = mode->getBitmap();
        if (bitmap.isReleased()) {
            bitmap.acquire();
            mode->forEachObject(&Object::acquireWithBitmap);
        }
#endif

        Object *obj = mode->getObject(addr + off);
        if (!obj) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->addr();
        // Handle objects with no memory in the accelerator
                ret = obj->copyToBuffer(buffer, c, bufferOff + off, objOff);
                obj->release();
        if(ret != gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
        off += objCount;
        TRACE(LOCAL,"Copying from obj %p: "FMT_SIZE" of "FMT_SIZE, obj->addr(), c, count);
    } while(addr + off < addr + count);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::fromIOBuffer(core::Mode &mode, hostptr_t addr, core::IOBuffer &buffer, size_t bufferOff, size_t count)
{
    if (count > (buffer.size() - bufferOff)) return gmacErrorInvalidSize;
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::Mode *mode = proc_.owner(addr + off);
        if (mode == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
#ifdef USE_VM
        CFATAL(mode->releasedObjects() == false, "Acquiring bitmap on released objects");
        vm::Bitmap &bitmap = mode->getBitmap();
        if (bitmap.isReleased()) {
            bitmap.acquire();
            mode->forEachObject(&Object::acquireWithBitmap);
        }
#endif
        Object *obj = mode->getObject(addr + off);
        if (!obj) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->addr();
                ret = obj->copyFromBuffer(buffer, c, bufferOff + off, objOff);
                obj->release();
        if(ret != gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
        off += objCount;
        TRACE(LOCAL,"Copying to obj %p: "FMT_SIZE" of "FMT_SIZE, obj->addr(), c, count);
    } while(addr + off < addr + count);
    trace::ExitCurrentFunction();
    return ret;
}

bool
Manager::signalRead(core::Mode &mode, hostptr_t addr)
{
    trace::EnterCurrentFunction();
#ifdef USE_VM
    CFATAL(mode.releasedObjects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = mode.getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        mode.forEachObject(&Object::acquireWithBitmap);
    }
#endif
    bool ret = true;
    Object *obj = mode.getObject(addr);
    if(obj == NULL) {
        trace::ExitCurrentFunction();
        return false;
    }
    TRACE(LOCAL,"Read access for object %p: %p", obj->addr(), addr);
    gmacError_t err = obj->signalRead(addr);
    ASSERTION(err == gmacSuccess);
    obj->release();
    trace::ExitCurrentFunction();
    return ret;
}

bool
Manager::signalWrite(core::Mode &mode, hostptr_t addr)
{
    trace::EnterCurrentFunction();
#ifdef USE_VM
    CFATAL(mode.releasedObjects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = mode.getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        mode.forEachObject(&Object::acquireWithBitmap);
    }
#endif
    bool ret = true;
    Object *obj = mode.getObject(addr);
    if(obj == NULL) {
        trace::ExitCurrentFunction();
        return false;
    }
    TRACE(LOCAL,"Write access for object %p: %p", obj->addr(), addr);
    if(obj->signalWrite(addr) != gmacSuccess) ret = false;
    obj->release();
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::memset(core::Mode &mode, hostptr_t s, int c, size_t size)
{
    trace::EnterCurrentFunction();
    core::Mode *owner = proc_.owner(s, size);
        if (owner == NULL) {
        ::memset(s, c, size);
        trace::ExitCurrentFunction();
        return gmacSuccess;
    }

#ifdef USE_VM
    CFATAL(owner->releasedObjects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = owner->getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        owner->forEachObject(&Object::acquireWithBitmap);
    }
#endif

    gmacError_t ret = gmacSuccess;

    Object *obj = owner->getObject(s);
    ASSERTION(obj != NULL);
    // Check for a fast path -- probably the user is just
    // initializing a single object or a portion of an object
    if(obj->addr() <= s && obj->end() >= (s + size)) {
        size_t objSize = (size < obj->size()) ? size : obj->size();
        ret = obj->memset(s - obj->addr(), c, objSize);
        obj->release();
        trace::ExitCurrentFunction();
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
        obj = owner->getObject(s);
    }

    trace::ExitCurrentFunction();
    return ret;
}

/**
 * Gets the number of bytes at the begining of a range that are in host memory
 * \param addr Starting address of the memory range
 * \param size Size (in bytes) of the memory range
 * \param obj First object within the range
 * \return Number of bytes at the beginning of the range that are in host memory
 */
static size_t
hostMemory(hostptr_t addr, size_t size, const Object *obj)
{
    // There is no object, so everything is in host memory
    if(obj == NULL) return size;

    // The object starts after the memory range, return the difference
    if(addr < obj->addr()) return obj->addr() - addr;

    ASSERTION(obj->end() > addr); // Sanity check
    return 0;
}

gmacError_t
Manager::memcpy(core::Mode &mode, hostptr_t dst, const hostptr_t src,
                size_t size)
{
    trace::EnterCurrentFunction();
    core::Mode *dstMode = proc_.owner(dst, size);
    core::Mode *srcMode = proc_.owner(src, size);

    if(dstMode == NULL && srcMode == NULL) {
        ::memcpy(dst, src, size);
        trace::ExitCurrentFunction();
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
            ret = srcObject->memcpyFromObject(mode, dst + offset, srcObjectOffset, copySize);
        }
        else if(srcHostMemory != 0) { // Host-to-object memory copy
            size_t dstCopySize = dstObject->end() - dst - offset;
            copySize = (srcHostMemory < dstCopySize) ? srcHostMemory : dstCopySize;
            size_t dstObjectOffset = dst + offset - dstObject->addr();
            ret = dstObject->memcpyToObject(mode, dstObjectOffset, src + offset, copySize);
        }
        else { // Object-to-object memory copy
            size_t srcCopySize = srcObject->end() - src - offset;
            size_t dstCopySize = dstObject->end() - dst - offset;
            copySize = (srcCopySize < dstCopySize) ? srcCopySize : dstCopySize;
            copySize = (copySize < left) ? copySize : left;
            size_t srcObjectOffset = src + offset - srcObject->addr();
            size_t dstObjectOffset = dst + offset - dstObject->addr();
            ret = srcObject->memcpyObjectToObject(mode, *dstObject, dstObjectOffset, srcObjectOffset, copySize);
        }

        offset += copySize;
        left -= copySize;
    }

    if(dstObject != NULL) dstObject->release();
    if(srcObject != NULL) srcObject->release();

    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::flushDirty(core::Mode &mode)
{
    gmacError_t ret;
    TRACE(LOCAL,"Flushing Objects");
    // Release per-mode objects
    ret = mode.protocol().flushDirty();

    if(ret == gmacSuccess) {
        // Release global per-process objects
        Protocol *protocol = proc_.protocol();
        if(protocol != NULL) protocol->flushDirty();
    }
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
