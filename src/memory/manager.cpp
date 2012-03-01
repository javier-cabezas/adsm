#include "memory/handler.h"
#include "memory/manager.h"
#include "memory/object.h"
#include "memory/object_mapped.h"
#include "memory/map_object.h"

#include "address_space.h"

namespace __impl { namespace memory {

list_addr AllAddresses;

manager::manager() :
    mapAllocations_("map_manager_allocations")
{
    TRACE(LOCAL,"Memory manager starts");
    Init();
    handler::setManager(*this);
}

manager::~manager()
{
}

host_ptr
manager::map(address_space_ptr aspace, host_ptr addr, size_t size,
             int flags, gmacError_t &err)
{
#if 0
    TRACE(LOCAL, "New mapping");
    trace::EnterCurrentFunction();

    host_ptr ret;
    if (aspace->is_integrated()) {
        ASSERTION(!addr, "NULL mappings not implemented");
        err = host_mapped_alloc(aspace, &ret, size);
        trace::ExitCurrentFunction();
        return NULL;
    }

    memory::map_object &map = aspace->get_object_map();

    if (!addr) {
        // Create new shared object. We set the memory as invalid to avoid stupid data transfers
        // to non-initialized objects
        object *newObject = map.get_protocol().create_object(NULL, size, GMAC_MAP_READWRITE, GMAC_MAP_READWRITE, err);
        if(err != gmacSuccess) {
            trace::ExitCurrentFunction();
            return NULL;
        }
        //newObject->add_mapping(aspace);
        ret = newObject->get_bounds().start;
    } else {
        object *obj = get_object(addr);

        obj->add_mapping(aspace);

        if (flagsHost == GMAC_MAP_NONE) {
            ASSERTION(flagsDevice != GMAC_MAP_NONE);

            // GPU-only object
        } else {
            if (flagsDevice == GMAC_MAP_NONE) {
                // Host-only object ->
            } else {
                // Shared object
            }
        }
    }

    trace::ExitCurrentFunction();
#endif
    return NULL;
}

#if 0
gmacError_t
manager::remap(address_space_ptr aspace, host_ptr old_addr, host_ptr *new_addr, size_t new_size, int flags)
{
    FATAL("MAP NOT IMPLEMENTED YET");
    gmacError_t ret = gmacSuccess;

    TRACE(LOCAL, "New remapping");
    trace::EnterCurrentFunction();

    return ret;
}
#endif

gmacError_t
manager::unmap(address_space_ptr aspace, host_ptr addr, size_t size)
{
    FATAL("UNMAP NOT IMPLEMENTED YET");
    TRACE(LOCAL, "Unmap allocation");
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::map_object &map = aspace->get_object_map();

    object_ptr ptr = map.get_object(addr);
    if (ptr)  {
        ptr->remove_owner(aspace);
        map.remove_object(*ptr);
    } else {
        object_mapped_ptr hostMappedObject = object_mapped::get_object(addr);
        if(hostMappedObject) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        object_mapped::remove(addr);
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t manager::alloc(address_space_ptr aspace, host_ptr *addr, size_t size)
{
    gmacError_t ret = gmacSuccess;

    TRACE(LOCAL, "New allocation");
    trace::EnterCurrentFunction();

    if (aspace->is_integrated()) {
        ret = host_mapped_alloc(aspace, addr, size);
        trace::ExitCurrentFunction();
        return ret;
    }

    memory::map_object &map = aspace->get_object_map();

    // Create new shared object. We set the memory as invalid to avoid stupid data transfers
    // to non-initialized objects
    object *newObject = map.get_protocol().create_object(NULL, size, GMAC_MAP_READWRITE, GMAC_MAP_READWRITE, ret);
    if (ret != gmacSuccess) {
        trace::ExitCurrentFunction();
        return ret;
    }
    newObject->add_owner(aspace);
    *addr = newObject->get_bounds().start;

    // Insert object into the global memory map
    std::pair<map_allocation::iterator, bool> res = mapAllocations_.insert(map_allocation::value_type(*addr + size, aspace));

    ASSERTION(res.second == true, "Object already registered");

    // Insert object into memory maps
    map.add_object(*newObject);
    trace::ExitCurrentFunction();
    return gmacSuccess;
}

gmacError_t manager::host_mapped_alloc(address_space_ptr aspace, host_ptr *addr, size_t size)
{
    TRACE(LOCAL, "New host-mapped allocation");
    trace::EnterCurrentFunction();
    object_mapped* obj = new object_mapped(aspace, size);
    *addr = obj->addr();
    if(*addr == NULL) {
        trace::ExitCurrentFunction();
        return gmacErrorMemoryAllocation;
    }
    trace::ExitCurrentFunction();
    return gmacSuccess;
}

#if 0
gmacError_t manager::globalAlloc(address_space_ptr aspace, host_ptr *addr, size_t size, GmacGlobalMallocType hint)
{
    TRACE(LOCAL, "New global allocation");
    trace::EnterCurrentFunction();

    // If a centralized object is requested, try creating it
    if(hint == GMAC_GLOBAL_MALLOC_CENTRALIZED) {
        gmacError_t ret = host_mapped_alloc(aspace, addr, size);
        if(ret == gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
    }
    protocol *protocol = proc_.get_protocol();
    if(protocol == NULL) return gmacErrorInvalidValue;
    object *object = protocol->create_object(size, NULL, GMAC_PROT_NONE, 0);
    *addr = object->addr();
    if(*addr == NULL) {
        object->decRef();
        trace::ExitCurrentFunction();
        return host_mapped_alloc(aspace, addr, size); // Try using a host mapped object
    }
    gmacError_t ret = proc_.globalMalloc(*object);
    object->decRef();
    trace::ExitCurrentFunction();
    return ret;
}
#endif

gmacError_t manager::free(address_space_ptr aspace, host_ptr addr)
{
    TRACE(LOCAL, "Free allocation");
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::map_object &map = aspace->get_object_map();

    object_ptr ptr = map.get_object(addr);
    if(ptr)  {
        map.remove_object(*ptr);

        map_allocation::iterator it = mapAllocations_.upper_bound(addr);
        ASSERTION(it != mapAllocations_.end(), "Object not registered");
        mapAllocations_.erase(it);
    } else {
        object_mapped_ptr hostMappedObject = object_mapped::get_object(addr);
        if(hostMappedObject) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        object_mapped::remove(addr);
    }
    trace::ExitCurrentFunction();
    return ret;
}

size_t
manager::get_alloc_size(address_space_ptr aspace, host_const_ptr addr, gmacError_t &err) const
{
    size_t ret = 0;
    trace::EnterCurrentFunction();
    err = gmacSuccess;

    memory::map_object &map = aspace->get_object_map();

    object_ptr obj = map.get_object(addr);
    if (!obj) {
        object_mapped_ptr hostMappedObject = object_mapped::get_object(addr);
        if (hostMappedObject) {
            ret = hostMappedObject->size();
        } else {
            err = gmacErrorInvalidValue;
        }
    } else {
        ret = obj->size();
    }
    trace::ExitCurrentFunction();
    return ret;
}

hal::ptr
manager::translate(address_space_ptr aspace, host_ptr addr)
{
    trace::EnterCurrentFunction();
    memory::map_object &map = aspace->get_object_map();

    object_ptr obj = map.get_object(addr);
    hal::ptr ret;
    if(!obj) {
        object_mapped_ptr object = object_mapped::get_object(addr);
        if(object) {
            ret = object->get_device_addr(aspace, addr);
        }
    } else {
        ret = obj->get_device_addr(addr);
    }
    trace::ExitCurrentFunction();
    return ret;
}

address_space_ptr
manager::get_owner(host_const_ptr addr, size_t size)
{
    address_space_ptr aspace;

    map_allocation::iterator it = mapAllocations_.upper_bound(addr);

    if (it != mapAllocations_.end()) {
        aspace = it->second;
    }

    return aspace;
}

gmacError_t
manager::acquire_objects(address_space_ptr aspace, const list_addr &addrs)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    hal::event_ptr evt;

    memory::map_object &map = aspace->get_object_map();
    if (addrs.size() == 0) {
        //if (map.released_objects()) {
        TRACE(LOCAL,"Acquiring Objects");
        GmacProtection prot = GMAC_PROT_READWRITE;
        evt = map.acquire_objects(prot, ret);
        map.acquire_objects();
        //}
    } else {
        TRACE(LOCAL,"Acquiring call Objects");
        std::list<object_access_info>::const_iterator it;
        for (it = addrs.begin(); it != addrs.end(); ++it) {
            object_ptr obj = map.get_object(it->first);
            if (!obj) {
                object_mapped_ptr hostMappedObject = object_mapped::get_object(it->first);
                ASSERTION(bool(hostMappedObject), "Address not found");
#ifdef USE_OPENCL
                hostMappedObject->acquire(aspace);
#endif
            } else {
                GmacProtection prot = it->second;
                evt = obj->acquire(prot, ret);
                ASSERTION(ret == gmacSuccess);
            }
        }
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
manager::release_objects(address_space_ptr aspace, const list_addr &addrs)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    hal::event_ptr evt;

    memory::map_object &map = aspace->get_object_map();
    if (addrs.size() == 0) { // Release all objects
        TRACE(LOCAL,"Releasing objects");
        if (map.has_modified_objects()) {
            // Mark objects as released
            evt = map.release_objects(false, ret);
            ASSERTION(ret == gmacSuccess);
            // Flush protocols
            // 1. Mode protocol
            evt = map.get_protocol().release_all(ret);
            ASSERTION(ret == gmacSuccess);

            map.release_objects();
        }
    } else { // Release given objects
        TRACE(LOCAL,"Releasing objects used in kernel call");
        list_addr::const_iterator it;
        for (it = addrs.begin(); it != addrs.end(); ++it) {
            object_ptr obj = map.get_object(it->first);
            if (!obj) {
                object_mapped_ptr hostMappedObject = object_mapped::get_object(it->first);
                ASSERTION(bool(hostMappedObject), "Address not found");
#ifdef USE_OPENCL
                hostMappedObject->release(aspace);
#endif
            } else {
                // Release all the blocks in the object
                evt = obj->release(true, ret);
                ASSERTION(ret == gmacSuccess);
            }
        }
    }
    trace::ExitCurrentFunction();
    return ret;
}

bool
manager::signal_read(address_space_ptr aspace, host_ptr addr)
{
    trace::EnterCurrentFunction();
    memory::map_object &map = aspace->get_object_map();

#ifdef USE_VM
    CFATAL(map.released_objects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = map.getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        map.for_each_object(&object::acquireWithBitmap);
    }
#endif

    bool ret = true;
    object_ptr obj = map.get_object(addr);
    if(!obj) {
        trace::ExitCurrentFunction();
        return false;
    }
    TRACE(LOCAL,"Read access for object %p: %p", obj->get_bounds().start, addr);
    gmacError_t err;
    obj->signal_read(addr, err);
    ASSERTION(err == gmacSuccess);
    trace::ExitCurrentFunction();
    return ret;
}

bool
manager::signal_write(address_space_ptr aspace, host_ptr addr)
{
    trace::EnterCurrentFunction();
    bool ret = true;
    memory::map_object &map = aspace->get_object_map();

#ifdef USE_VM
    CFATAL(map.released_objects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = map.getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        map.for_each_object(&object::acquireWithBitmap);
    }
#endif

    object_ptr obj = map.get_object(addr);
    if(!obj) {
        trace::ExitCurrentFunction();
        return false;
    }
    TRACE(LOCAL,"Write access for object %p: %p", obj->get_bounds().start, addr);
    gmacError_t err;
    obj->signal_write(addr, err);
    if (err != gmacSuccess) ret = false;
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
manager::memset(address_space_ptr aspace, host_ptr s, int c, size_t size)
{
    trace::EnterCurrentFunction();
    address_space_ptr aspaceOwner = get_owner(s, size);

    if (!aspaceOwner) {
        ::memset(s, c, size);
        trace::ExitCurrentFunction();
        return gmacSuccess;
    }

#ifdef USE_VM
    CFATAL(aspaceOwner->released_objects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = aspaceOwner->getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        aspaceOwner->for_each_object(&object::acquireWithBitmap);
    }
#endif

    gmacError_t ret = gmacSuccess;

    memory::map_object &map = aspaceOwner->get_object_map();
    object_ptr obj = map.get_object(s);
    ASSERTION(bool(obj));
    // Check for a fast path -- probably the user is just
    // initializing a single object or a portion of an object
    if(obj->get_bounds().start <= s && obj->get_bounds().end >= (s + size)) {
        size_t objSize = (size < obj->size()) ? size : obj->size();
        ret = obj->memset(s - obj->get_bounds().start, c, objSize);
        trace::ExitCurrentFunction();
        return ret;
    }

    // This code handles the case of the user initializing a portion of
    // memory that includes host memory and GMAC objects.
    size_t left = size;
    while(left > 0) {
        // If there is no object, initialize the remaining host memory
        if(!obj) {
            ::memset(s, c, left);
            left = 0; // This will finish the loop
        } else {
            // Check if there is a memory gap of host memory at the begining of the
            // memory range that remains to be initialized
            int gap = int(obj->get_bounds().start - s);
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
            ret = obj->memset(s - obj->get_bounds().start, c, objSize);
            if(ret != gmacSuccess) break;
            left -= objSize; // Account for the bytes initialized by the object
            s += objSize;  // Advance the pointer
        }
        // Get the next object in the memory range that remains to be initialized
        if (left > 0) {
            memory::map_object &map = aspaceOwner->get_object_map();
            obj = map.get_object(s);
        }
    }

    trace::ExitCurrentFunction();
    return ret;
}

/**
 * Gets the number of bytes at the beginning of a range that are in host memory
 * \param addr Starting address of the memory range
 * \param size Size (in bytes) of the memory range
 * \param obj First object within the range
 * \return Number of bytes at the beginning of the range that are in host memory
 */
static size_t
get_host_memory_size_at_start(host_const_ptr addr, size_t size, const object *obj)
{
    // There is no object, so everything is in host memory
    if(obj == NULL) return size;

    // The object starts after the memory range, return the difference
    if(addr < obj->get_bounds().start) return obj->get_bounds().start - addr;

    ASSERTION(obj->get_bounds().end > addr); // Sanity check
    return 0;
}

gmacError_t
manager::memcpy(address_space_ptr aspace, host_ptr dst, host_const_ptr src,
                size_t size)
{
    trace::EnterCurrentFunction();
    address_space_ptr aspaceDst = get_owner(dst, size);
    address_space_ptr aspaceSrc = get_owner(src, size);

    if(!aspaceDst && !aspaceSrc) {
        ::memcpy(dst, src, size);
        trace::ExitCurrentFunction();
        return gmacSuccess;
    }

    object_ptr dstObject;
    object_ptr srcObject;

    memory::map_object *mapDst = NULL;
    memory::map_object *mapSrc = NULL;

    // Get initial objects
    if(aspaceDst) {
        mapDst = &aspaceDst->get_object_map();
        dstObject = mapDst->get_object(dst, size);
    }
    if(aspaceSrc) {
        mapSrc = &aspaceSrc->get_object_map();
        srcObject = mapSrc->get_object(src, size);
    }

    gmacError_t ret = gmacSuccess;
    size_t left = size;
    size_t offset = 0;
    size_t copySize = 0;
    while(left > 0) {
        // Get next objects involved, if necessary
        if(aspaceDst && dstObject && dstObject->get_bounds().end < (dst + offset)) {
            dstObject = mapDst->get_object(dst + offset, left);
        }
        if(aspaceSrc && srcObject && srcObject->get_bounds().end < (src + offset)) {
            srcObject = mapSrc->get_object(src + offset, left);
        }

        // Get the number of host-to-host memory we have to copy
        size_t dstHostMemory = get_host_memory_size_at_start(dst + offset, left, dstObject.get());
        size_t srcHostMemory = get_host_memory_size_at_start(src + offset, left, srcObject.get());

        if(dstHostMemory != 0 && srcHostMemory != 0) { // Host-to-host memory copy
            copySize = (dstHostMemory < srcHostMemory) ? dstHostMemory : srcHostMemory;
            ::memcpy(dst + offset, src + offset, copySize);
            ret = gmacSuccess;
        }
        else if(dstHostMemory != 0) { // Object-to-host memory copy
            size_t srcCopySize = srcObject->get_bounds().end - src - offset;
            copySize = (dstHostMemory < srcCopySize) ? dstHostMemory : srcCopySize;
            size_t srcObjectOffset = src + offset - srcObject->get_bounds().start;
            ret = srcObject->memcpy_from_object(dst + offset, srcObjectOffset, copySize);
        }
        else if(srcHostMemory != 0) { // Host-to-object memory copy
            size_t dstCopySize = dstObject->get_bounds().end - dst - offset;
            copySize = (srcHostMemory < dstCopySize) ? srcHostMemory : dstCopySize;
            size_t dstObjectOffset = dst + offset - dstObject->get_bounds().start;
            ret = dstObject->memcpy_to_object(dstObjectOffset, src + offset, copySize);
        }
        else { // Object-to-object memory copy
            size_t srcCopySize = srcObject->get_bounds().end - src - offset;
            size_t dstCopySize = dstObject->get_bounds().end - dst - offset;
            copySize = (srcCopySize < dstCopySize) ? srcCopySize : dstCopySize;
            copySize = (copySize < left) ? copySize : left;
            size_t srcObjectOffset = src + offset - srcObject->get_bounds().start;
            size_t dstObjectOffset = dst + offset - dstObject->get_bounds().start;
            ret = srcObject->memcpy_object_to_object(*dstObject, dstObjectOffset, srcObjectOffset, copySize);
        }

        offset += copySize;
        left -= copySize;
    }

    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
manager::flush_dirty(address_space_ptr aspace)
{
    gmacError_t ret;
    hal::event_ptr evt;
    TRACE(LOCAL,"Flushing Objects");
    // Release per-mode objects
    memory::map_object &map = aspace->get_object_map();
    evt = map.get_protocol().flush_dirty(ret);

    return ret;
}

gmacError_t
manager::from_io_device(address_space_ptr aspace, host_ptr addr,
                        hal::device_input &input, size_t count)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        address_space_ptr aspace = get_owner(addr + off);
        if (!aspace) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        memory::map_object &map = aspace->get_object_map();
        object_ptr obj = map.get_object(addr + off);
        if (!obj) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        // Compute sizes for the current object
        size_t objCount = obj->get_bounds().start + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->get_bounds().start;
        ret = obj->from_io_device(objOff, input, c);
        if(ret != gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
        off += objCount;
        TRACE(LOCAL,"Copying to obj %p: "FMT_SIZE" of "FMT_SIZE, obj->get_bounds().start, c, count);
    } while(addr + off < addr + count);
    trace::ExitCurrentFunction();
    return ret;

}

gmacError_t
manager::to_io_device(hal::device_output &output, address_space_ptr aspace, host_const_ptr addr, size_t count)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        address_space_ptr aspace = get_owner(addr + off);
        if (!aspace) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        memory::map_object &map = aspace->get_object_map();
        object_ptr obj = map.get_object(addr + off);
        if (!obj) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        // Compute sizes for the current object
        size_t objCount = obj->get_bounds().start + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->get_bounds().start;
        // Handle objects with no memory in the accelerator
        ret = obj->to_io_device(output, objOff, c);
        if(ret != gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
        off += objCount;
        TRACE(LOCAL,"Copying from obj %p: "FMT_SIZE" of "FMT_SIZE, obj->get_bounds().start, c, count);
    } while(addr + off < addr + count);
    trace::ExitCurrentFunction();
    return ret;

}

#if 0
gmacError_t manager::moveTo(host_ptr addr, address_space_ptr aspace)
{
    object * obj = mode.getObjectWrite(addr);
    if(obj == NULL) return gmacErrorInvalidValue;

    mode.putObject(*obj);
#if 0
    StateObject<T>::lock_write();
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
