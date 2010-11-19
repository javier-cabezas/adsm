#if defined(__GNUC__)
#include <strings.h>
#elif defined(_MSC_VER)
#define strcasecmp _stricmp
#endif

#include "core/IOBuffer.h"
#include "core/Process.h"
#include "protocol/Lazy.h"

#include "Manager.h"
#include "Map.h"
#include "Object.h"

namespace __impl { namespace memory {

Manager::Manager()
{
    TRACE(LOCAL,"Memory manager starts");
    // Create protocol
    if(strcasecmp(paramProtocol, "Rolling") == 0) {
        protocol_ = new protocol::Lazy((unsigned)paramRollSize);
    }
    else if(strcasecmp(paramProtocol, "Lazy") == 0) {
        protocol_ = new protocol::Lazy((unsigned)-1);
    }
    else {
        FATAL("Memory Coherence Protocol not defined");
    }
}

Manager::~Manager()
{
    TRACE(LOCAL,"Memory manager finishes");
    delete protocol_;
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
    Object *object = protocol_->createObject(size, NULL, GMAC_PROT_READ);
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

#if 0
gmacError_t Manager::globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint)
{
    core::Process &proc = core::Process::getInstance();
    core::Mode &mode = core::Mode::current();

    if (proc.allIntegrated()) hint = GMAC_GLOBAL_MALLOC_CENTRALIZED;
    if(hint == GMAC_GLOBAL_MALLOC_REPLICATED) {
        Object *object = protocol_->createReplicatedObject(size);
        *addr = object->addr();
        if(*addr == NULL) {
            delete object;
            return gmacErrorMemoryAllocation;
        }

        mode.addReplicatedObject(*object);
    }
    else /*if (GMAC_GLOBAL_MALLOC_CENTRALIZED)*/ {
        Object *object = protocol_->createCentralizedObject(size);
        *addr = object->addr();
        if(*addr == NULL) {
            delete object;
            return gmacErrorMemoryAllocation;
        }

        mode.addCentralizedObject(*object);
#if 0
    } else {
        return gmacErrorInvalidValue;
#endif
    }

    return gmacSuccess;
}


#endif

gmacError_t Manager::free(void * addr)
{
    gmacError_t ret = gmacSuccess;
    core::Mode &mode = core::Mode::current();
    const Object *object = mode.getObject(addr);
    if(object != NULL)  {
        mode.removeObject(*object);
		object->release();
    }
    else ret = gmacErrorInvalidValue;
    return ret;
}

gmacError_t Manager::acquire()
{
    gmacError_t ret = gmacSuccess;
    core::Mode &mode = core::Mode::current();
    if (mode.releasedObjects() == false) {
        return gmacSuccess;
    }
	mode.forEachObject(&Object::acquire);
    mode.releaseObjects();
    return ret;
}

gmacError_t Manager::release()
{
#ifdef USE_VM
    checkBitmapToDevice();
#endif
    core::Mode &mode = core::Mode::current();
    TRACE(LOCAL,"Releasing Objects");
    gmacError_t ret = gmacSuccess;
    ret = protocol_->release();

    mode.releaseObjects();
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
    core::Mode *mode = core::Mode::current();
    vm::Bitmap &bitmap = mode->dirtyBitmap();
    if (!bitmap.synced()) {
        bitmap.syncHost();

        const Map &map = mode->objects();
        map.lockRead();
        Map::const_iterator i;
        for(i = map.begin(); i != map.end(); i++) {
            Object &object = *i->second;
            gmacError_t ret = protocol_->acquireWithBitmap(object);
            ASSERTION(ret == gmacSuccess);
        }
        map.unlock();
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
#if 0
gmacError_t Manager::memcpy(void * dst, const void * src, size_t n)
{
    core::Process &proc = core::Process::getInstance();
    core::Mode *dstMode = proc.owner(dst);
    core::Mode *srcMode = proc.owner(src);

	if (dstMode == NULL && srcMode == NULL) {
        ::memcpy(dst, src, n);
        return gmacSuccess;
    }

    const Object *dstObj = NULL;
    const Object *srcObj = NULL;
	if (dstMode != NULL) {
        dstObj = dstMode->getObjectRead(dst);
        ASSERTION(dstObj != NULL);
    }
    if (srcMode != NULL) {
        srcObj = srcMode->getObjectRead(src);
        ASSERTION(srcObj != NULL);
    }

    gmacError_t err = gmacSuccess;
    if (dstMode == NULL) {	    // From device
		err = protocol_->toPointer(dst, *srcObj, (unsigned)((uint8_t *)src - srcObj->addr()), n);
        ASSERTION(err == gmacSuccess);
	}
    else if(srcMode == NULL) {   // To device
		err = protocol_->fromPointer(*dstObj, (unsigned)((uint8_t *)dst - dstObj->addr()), src, n);
        ASSERTION(err == gmacSuccess);
    }
    else {
        if (!srcObj->isInAccelerator() && !dstObj->isInAccelerator()) {
            ::memcpy(dst, src, n);
        } else if (srcObj->isInAccelerator() && !dstObj->isInAccelerator()) {
            err = protocol_->toPointer(dst, *srcObj, (unsigned)((uint8_t *)src - srcObj->addr()), n);
            ASSERTION(err == gmacSuccess);
        } else if (!srcObj->isInAccelerator() && dstObj->isInAccelerator()) {
            err = protocol_->fromPointer(*dstObj, (unsigned)((uint8_t *)dst - dstObj->addr()), src, n);
            ASSERTION(err == gmacSuccess);
        } else {
            err = protocol_->copy(*dstObj, (unsigned)((uint8_t *)dst - dstObj->addr()),
                                  *srcObj, (unsigned)((uint8_t *)src - srcObj->addr()), n);
            ASSERTION(err == gmacSuccess);
        }
	}

    if (dstMode != NULL) {
        dstMode->putObject(*dstObj);
    }
    if (srcMode != NULL) {
        srcMode->putObject(*srcObj);
    }
#if 0
	else { // dstCtx != srcCtx
        core::Mode *mode = core::Mode::current();
        ASSERTION(mode != NULL);

        manager->release((void *)src, n);
        manager->invalidate(dst, n);

        off_t off = 0;
        core::IOBuffer *buffer = core::Mode::current()->getIOBuffer();

        size_t left = n;
        while (left != 0) {
            size_t bytes = left < buffer->size() ? left : buffer->size();
            err = srcMode->bufferToHost(buffer, proc->translate((char *)src + off), bytes);
            ASSERTION(err == gmacSuccess);

            err = dstMode->bufferToDevice(buffer, proc->translate((char *)dst + off), bytes);
            ASSERTION(err == gmacSuccess);

            left -= bytes;
            off  += bytes;
        }

	}
#endif
    return err;
}

gmacError_t Manager::memset(void *s, int c, size_t n)
{
    core::Process &proc = core::Process::getInstance();
    core::Mode *mode = proc.owner(s);
	if (mode == NULL) {
        ::memset(s, c, n);
        return gmacSuccess;
    }

    const Object * obj = mode->getObjectRead(s);
    ASSERTION(obj != NULL);
	if (obj->isInAccelerator() == false) {
        ::memset(s, c, n);
        return gmacSuccess;
    }
    gmacError_t ret;
    ret = protocol_->memset(*obj, unsigned((uint8_t *)s - obj->addr()), c, n);
    mode->putObject(*obj);
    return ret;
}

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
