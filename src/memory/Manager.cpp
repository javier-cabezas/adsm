#include "core/address_space.h"
#include "core/process.h"

#include "memory/Handler.h"
#include "memory/HostMappedObject.h"
#include "memory/Manager.h"
#include "memory/object.h"
#include "memory/map_object.h"

using __impl::util::params::ParamAutoSync;


namespace __impl { namespace memory {

ListAddr AllAddresses;

Manager::Manager(core::process &proc) :
    proc_(proc),
    mapAllocations_("map_manager_allocations")
{
    TRACE(LOCAL,"Memory manager starts");
    Init();
    Handler::setManager(*this);
}

Manager::~Manager()
{
}

gmacError_t
Manager::map(core::address_space_ptr aspace, hostptr_t *addr, size_t size, int flags)
{
    FATAL("MAP NOT IMPLEMENTED YET");
    gmacError_t ret = gmacSuccess;

    TRACE(LOCAL, "New mapping");
    trace::EnterCurrentFunction();
    // For integrated accelerators we want to use Centralized objects to avoid memory transfers
    // TODO: ask process instead
    // if (mode.getAccelerator().integrated()) return hostMappedAlloc(addr, size);

    memory::map_object &map = aspace->get_object_map();

    object *object;
    if (*addr != NULL) {
        object = map.getObject(*addr);
        if(object != NULL) {
            // TODO: Remove this limitation
            ASSERTION(object->size() == size);
            ret = object->addOwner(aspace);
            goto done;
        }
    }

    // Create new shared object. We set the memory as invalid to avoid stupid data transfers
    // to non-initialized objects
    object = map.getProtocol().createObject(size, NULL, GMAC_PROT_READ, 0);
    if(object == NULL) {
        trace::ExitCurrentFunction();
        return gmacErrorMemoryAllocation;
    }
    object->addOwner(aspace);
    *addr = object->addr();

    // Insert object into memory maps
    map.addObject(*object);

done:
    object->decRef();
    trace::ExitCurrentFunction();
    return ret;
}


gmacError_t
Manager::remap(core::address_space_ptr aspace, hostptr_t old_addr, hostptr_t *new_addr, size_t new_size, int flags)
{
    FATAL("MAP NOT IMPLEMENTED YET");
    gmacError_t ret = gmacSuccess;

    TRACE(LOCAL, "New remapping");
    trace::EnterCurrentFunction();

    return ret;
}

gmacError_t
Manager::unmap(core::address_space_ptr aspace, hostptr_t addr, size_t size)
{
    FATAL("UNMAP NOT IMPLEMENTED YET");
    TRACE(LOCAL, "Unmap allocation");
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::map_object &map = aspace->get_object_map();

    object *object = map.getObject(addr);
    if(object != NULL)  {
        object->removeOwner(aspace);
        map.removeObject(*object);
        object->decRef();
    } else {
        HostMappedObject *hostMappedObject = HostMappedObject::get(addr);
        if(hostMappedObject == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        hostMappedObject->decRef();
        // We need to release the object twice to effectively destroy it
        HostMappedObject::remove(addr);
        hostMappedObject->decRef();
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::alloc(core::address_space_ptr aspace, hostptr_t *addr, size_t size)
{
    TRACE(LOCAL, "New allocation");
    trace::EnterCurrentFunction();
    // For integrated accelerators we want to use Centralized objects to avoid memory transfers
    // TODO: ask process instead
    if (aspace->is_integrated()) {
        gmacError_t ret = hostMappedAlloc(aspace, addr, size);
        trace::ExitCurrentFunction();
        return ret;
    }

    memory::map_object &map = aspace->get_object_map();

    // Create new shared object. We set the memory as invalid to avoid stupid data transfers
    // to non-initialized objects
    object *object = map.getProtocol().createObject(size, NULL, GMAC_PROT_READ, 0);
    if(object == NULL) {
        trace::ExitCurrentFunction();
        return gmacErrorMemoryAllocation;
    }
    object->addOwner(aspace);
    *addr = object->addr();

    // Insert object into the global memory map
    ASSERTION(mapAllocations_.find(*addr) == mapAllocations_.end(), "Object already registered");
    mapAllocations_.insert(map_allocation::value_type(*addr + size, aspace));

    // Insert object into memory maps
    map.addObject(*object);
    object->decRef();
    trace::ExitCurrentFunction();
    return gmacSuccess;
}

gmacError_t Manager::hostMappedAlloc(core::address_space_ptr aspace, hostptr_t *addr, size_t size)
{
    TRACE(LOCAL, "New host-mapped allocation");
    trace::EnterCurrentFunction();
    HostMappedObject *object = new HostMappedObject(aspace, size);
    *addr = object->addr();
    if(*addr == NULL) {
        object->decRef();
        trace::ExitCurrentFunction();
        return gmacErrorMemoryAllocation;
    }
    trace::ExitCurrentFunction();
    return gmacSuccess;
}

#if 0
gmacError_t Manager::globalAlloc(core::address_space_ptr aspace, hostptr_t *addr, size_t size, GmacGlobalMallocType hint)
{
    TRACE(LOCAL, "New global allocation");
    trace::EnterCurrentFunction();

    // If a centralized object is requested, try creating it
    if(hint == GMAC_GLOBAL_MALLOC_CENTRALIZED) {
        gmacError_t ret = hostMappedAlloc(aspace, addr, size);
        if(ret == gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
    }
    Protocol *protocol = proc_.getProtocol();
    if(protocol == NULL) return gmacErrorInvalidValue;
    object *object = protocol->createObject(size, NULL, GMAC_PROT_NONE, 0);
    *addr = object->addr();
    if(*addr == NULL) {
        object->decRef();
        trace::ExitCurrentFunction();
        return hostMappedAlloc(aspace, addr, size); // Try using a host mapped object
    }
    gmacError_t ret = proc_.globalMalloc(*object);
    object->decRef();
    trace::ExitCurrentFunction();
    return ret;
}
#endif

gmacError_t Manager::free(core::address_space_ptr aspace, hostptr_t addr)
{
    TRACE(LOCAL, "Free allocation");
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::map_object &map = aspace->get_object_map();

    object *object = map.getObject(addr);
    if(object != NULL)  {
        map.removeObject(*object);
        object->decRef();

        map_allocation::iterator it = mapAllocations_.upper_bound(addr);
        ASSERTION(it != mapAllocations_.end(), "Object not registered");
        mapAllocations_.erase(it);
    } else {
        HostMappedObject *hostMappedObject = HostMappedObject::get(addr);
        if(hostMappedObject == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        hostMappedObject->decRef();
        // We need to release the object twice to effectively destroy it
        HostMappedObject::remove(addr);
        hostMappedObject->decRef();
    }
    trace::ExitCurrentFunction();
    return ret;
}

size_t
Manager::getAllocSize(core::address_space_ptr aspace, const hostptr_t addr, gmacError_t &err) const
{
    size_t ret;
    trace::EnterCurrentFunction();
    err = gmacSuccess;

    memory::map_object &map = aspace->get_object_map();

    object *obj = map.getObject(addr);
    if (obj == NULL) {
        HostMappedObject *hostMappedObject = HostMappedObject::get(addr);
        if (hostMappedObject != NULL) {
            ret = hostMappedObject->size();
        } else {
            err = gmacErrorInvalidValue;
        }
    } else {
        ret = obj->size();
        obj->decRef();
    }
    trace::ExitCurrentFunction();
    return ret;
}

accptr_t
Manager::translate(core::address_space_ptr aspace, const hostptr_t addr)
{
    trace::EnterCurrentFunction();
    memory::map_object &map = aspace->get_object_map();

    object *obj = map.getObject(addr);
    accptr_t ret(0);
    if(obj == NULL) {
        HostMappedObject *object = HostMappedObject::get(addr);
        if(object != NULL) {
            ret = object->get_device_addr(aspace, addr);
            object->decRef();
        }
    } else {
        ret = obj->get_device_addr(addr);
        obj->decRef();
    }
    trace::ExitCurrentFunction();
    return ret;
}

core::address_space_ptr
Manager::owner(hostptr_t addr, size_t size)
{
    core::address_space_ptr aspace;

    map_allocation::iterator it = mapAllocations_.upper_bound(addr);

    if (it != mapAllocations_.end()) {
        aspace = it->second;
    }

    return aspace;
}

gmacError_t
Manager::acquireObjects(core::address_space_ptr aspace, const ListAddr &addrs)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::map_object &map = aspace->get_object_map();
    if (addrs.size() == 0) {
        if (map.hasModifiedObjects() && map.releasedObjects()) {
            TRACE(LOCAL,"Acquiring Objects");
            GmacProtection prot = GMAC_PROT_READWRITE;
            ret = map.forEachObject<GmacProtection>(&object::acquire, prot);
            map.acquireObjects();
        }
    } else {
        TRACE(LOCAL,"Acquiring call Objects");
        std::list<ObjectInfo>::const_iterator it;
        for (it = addrs.begin(); it != addrs.end(); it++) {
            object *obj = map.getObject(it->first);
            if (obj == NULL) {
                HostMappedObject *hostMappedObject = HostMappedObject::get(it->first);
                ASSERTION(hostMappedObject != NULL, "Address not found");
#ifdef USE_OPENCL
                hostMappedObject->acquire(aspace);
#endif
                hostMappedObject->decRef();
            } else {
                GmacProtection prot = it->second;
                ret = obj->acquire(prot);
                ASSERTION(ret == gmacSuccess);
                obj->decRef();
            }
        }
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::releaseObjects(core::address_space_ptr aspace, const ListAddr &addrs)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::map_object &map = aspace->get_object_map();
    if (addrs.size() == 0) { // Release all objects
        TRACE(LOCAL,"Releasing Objects");
        if (map.hasModifiedObjects()) {
            // Mark objects as released
            ret = map.forEachObject(&object::release);
            ASSERTION(ret == gmacSuccess);
            // Flush protocols
            // 1. Mode protocol
            ret = map.getProtocol().releaseAll();
            ASSERTION(ret == gmacSuccess);

            map.releaseObjects();
        }
    } else { // Release given objects
        TRACE(LOCAL,"Releasing call Objects");
        ListAddr::const_iterator it;
        for (it = addrs.begin(); it != addrs.end(); it++) {
            object *obj = map.getObject(it->first);
            if (obj == NULL) {
                HostMappedObject *hostMappedObject = HostMappedObject::get(it->first);
                ASSERTION(hostMappedObject != NULL, "Address not found");
#ifdef USE_OPENCL
                hostMappedObject->release(aspace);
#endif
                hostMappedObject->decRef();
            } else {
                // Release all the blocks in the object
                ret = obj->releaseBlocks();
                ASSERTION(ret == gmacSuccess);
                obj->decRef();
            }
        }

        // Notify protocols
        // 1. Mode protocol
        ret = map.getProtocol().releasedAll();
        ASSERTION(ret == gmacSuccess);

        map.releaseObjects();
    }
    trace::ExitCurrentFunction();
    return ret;
}

#if 0
gmacError_t Manager::toIOBuffer(core::address_space_ptr current, core::io_buffer &buffer, size_t bufferOff, const hostptr_t addr, size_t count)
{
    if (count > (buffer.size() - bufferOff)) return gmacErrorInvalidSize;
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::address_space_ptr aspace = owner(addr + off);
        if (aspace == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }

#ifdef USE_VM
        CFATAL(aspace->releasedObjects() == false, "Acquiring bitmap on released objects");
        vm::Bitmap &bitmap = aspace->getBitmap();
        if (bitmap.isReleased()) {
            bitmap.acquire();
            aspace->forEachObject(&object::acquireWithBitmap);
        }
#endif

        memory::map_object &map = aspace->get_object_map();
        object *obj = map.getObject(addr + off);
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
        obj->decRef();
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

gmacError_t Manager::fromIOBuffer(core::address_space_ptr current, hostptr_t addr, core::io_buffer &buffer, size_t bufferOff, size_t count)
{
    if (count > (buffer.size() - bufferOff)) return gmacErrorInvalidSize;
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::address_space_ptr aspace = owner(addr + off);
        if (aspace == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
#ifdef USE_VM
        CFATAL(aspace->releasedObjects() == false, "Acquiring bitmap on released objects");
        vm::Bitmap &bitmap = mode->getBitmap();
        if (bitmap.isReleased()) {
            bitmap.acquire();
            mode->forEachObject(&object::acquireWithBitmap);
        }
#endif
        memory::map_object &map = aspace->get_object_map();
        object *obj = map.getObject(addr + off);
        if (!obj) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->addr();
        ret = obj->copyFromBuffer(buffer, c, bufferOff + off, objOff);
        obj->decRef();
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
#endif

bool
Manager::signalRead(core::address_space_ptr aspace, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    memory::map_object &map = aspace->get_object_map();

#ifdef USE_VM
    CFATAL(map.releasedObjects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = map.getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        map.forEachObject(&object::acquireWithBitmap);
    }
#endif

    bool ret = true;
    object *obj = map.getObject(addr);
    if(obj == NULL) {
        trace::ExitCurrentFunction();
        return false;
    }
    TRACE(LOCAL,"Read access for object %p: %p", obj->addr(), addr);
    gmacError_t err = obj->signalRead(addr);
    ASSERTION(err == gmacSuccess);
    obj->decRef();
    trace::ExitCurrentFunction();
    return ret;
}

bool
Manager::signalWrite(core::address_space_ptr aspace, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    bool ret = true;
    memory::map_object &map = aspace->get_object_map();

#ifdef USE_VM
    CFATAL(map.releasedObjects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = map.getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        map.forEachObject(&object::acquireWithBitmap);
    }
#endif

    object *obj = map.getObject(addr);
    if(obj == NULL) {
        trace::ExitCurrentFunction();
        return false;
    }
    TRACE(LOCAL,"Write access for object %p: %p", obj->addr(), addr);
    if(obj->signalWrite(addr) != gmacSuccess) ret = false;
    obj->decRef();
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::memset(core::address_space_ptr aspace, hostptr_t s, int c, size_t size)
{
    trace::EnterCurrentFunction();
    core::address_space_ptr aspaceOwner = owner(s, size);

    if (aspaceOwner == NULL) {
        ::memset(s, c, size);
        trace::ExitCurrentFunction();
        return gmacSuccess;
    }

#ifdef USE_VM
    CFATAL(aspaceOwner->releasedObjects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = aspaceOwner->getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        aspaceOwner->forEachObject(&object::acquireWithBitmap);
    }
#endif

    gmacError_t ret = gmacSuccess;

    memory::map_object &map = aspaceOwner->get_object_map();
    object *obj = map.getObject(s);
    ASSERTION(obj != NULL);
    // Check for a fast path -- probably the user is just
    // initializing a single object or a portion of an object
    if(obj->addr() <= s && obj->end() >= (s + size)) {
        size_t objSize = (size < obj->size()) ? size : obj->size();
        ret = obj->memset(s - obj->addr(), c, objSize);
        obj->decRef();
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
        } else {
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
            obj->decRef();  // Release the object (it will not be needed anymore)
        }
        // Get the next object in the memory range that remains to be initialized
        if (left > 0) {
            memory::map_object &map = aspaceOwner->get_object_map();
            obj = map.getObject(s);
        }
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
hostMemory(hostptr_t addr, size_t size, const object *obj)
{
    // There is no object, so everything is in host memory
    if(obj == NULL) return size;

    // The object starts after the memory range, return the difference
    if(addr < obj->addr()) return obj->addr() - addr;

    ASSERTION(obj->end() > addr); // Sanity check
    return 0;
}

gmacError_t
Manager::memcpy(core::address_space_ptr aspace, hostptr_t dst, const hostptr_t src,
                size_t size)
{
    trace::EnterCurrentFunction();
    core::address_space_ptr aspaceDst = owner(dst, size);
    core::address_space_ptr aspaceSrc = owner(src, size);

    if(aspaceDst == NULL && aspaceSrc == NULL) {
        ::memcpy(dst, src, size);
        trace::ExitCurrentFunction();
        return gmacSuccess;
    }

    object *dstObject = NULL;
    object *srcObject = NULL;

    memory::map_object *mapDst = NULL;
    memory::map_object *mapSrc = NULL;

    // Get initial objects
    if(aspaceDst != NULL) {
        mapDst = &aspaceDst->get_object_map();
        dstObject = mapDst->getObject(dst, size);
    }
    if(aspaceSrc != NULL) {
        mapSrc = &aspaceSrc->get_object_map();
        srcObject = mapSrc->getObject(src, size);
    }

    gmacError_t ret = gmacSuccess;
    size_t left = size;
    size_t offset = 0;
    size_t copySize = 0;
    while(left > 0) {
        // Get next objects involved, if necessary
        if(aspaceDst != NULL && dstObject != NULL && dstObject->end() < (dst + offset)) {
            dstObject->decRef();
            dstObject = mapDst->getObject(dst + offset, left);
        }
        if(aspaceSrc != NULL && srcObject != NULL && srcObject->end() < (src + offset)) {
            srcObject->decRef();
            srcObject = mapSrc->getObject(src + offset, left);
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
            ret = srcObject->memcpyFromObject(dst + offset, srcObjectOffset, copySize);
        }
        else if(srcHostMemory != 0) { // Host-to-object memory copy
            size_t dstCopySize = dstObject->end() - dst - offset;
            copySize = (srcHostMemory < dstCopySize) ? srcHostMemory : dstCopySize;
            size_t dstObjectOffset = dst + offset - dstObject->addr();
            ret = dstObject->memcpyToObject(dstObjectOffset, src + offset, copySize);
        }
        else { // Object-to-object memory copy
            size_t srcCopySize = srcObject->end() - src - offset;
            size_t dstCopySize = dstObject->end() - dst - offset;
            copySize = (srcCopySize < dstCopySize) ? srcCopySize : dstCopySize;
            copySize = (copySize < left) ? copySize : left;
            size_t srcObjectOffset = src + offset - srcObject->addr();
            size_t dstObjectOffset = dst + offset - dstObject->addr();
            ret = srcObject->memcpyObjectToObject(*dstObject, dstObjectOffset, srcObjectOffset, copySize);
        }

        offset += copySize;
        left -= copySize;
    }

    if(dstObject != NULL) dstObject->decRef();
    if(srcObject != NULL) srcObject->decRef();

    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::flushDirty(core::address_space_ptr aspace)
{
    gmacError_t ret;
    TRACE(LOCAL,"Flushing Objects");
    // Release per-mode objects
    memory::map_object &map = aspace->get_object_map();
    ret = map.getProtocol().flushDirty();

    return ret;
}

gmacError_t
Manager::from_io_device(core::address_space_ptr aspace, hostptr_t addr,
                        hal::device_input &input, size_t count)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::address_space_ptr aspace = owner(addr + off);
        if (aspace == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        memory::map_object &map = aspace->get_object_map();
        object *obj = map.getObject(addr + off);
        if (!obj) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->addr();
        ret = obj->from_io_device(objOff, input, c);
        obj->decRef();
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

gmacError_t
Manager::to_io_device(hal::device_output &output, core::address_space_ptr aspace, const hostptr_t addr, size_t count)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::address_space_ptr aspace = owner(addr + off);
        if (aspace) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        memory::map_object &map = aspace->get_object_map();
        object *obj = map.getObject(addr + off);
        if (!obj) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->addr();
        // Handle objects with no memory in the accelerator
        ret = obj->to_io_device(output, objOff, c);
        obj->decRef();
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

#if 0
gmacError_t Manager::moveTo(hostptr_t addr, core::address_space_ptr aspace)
{
    object * obj = mode.getObjectWrite(addr);
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
