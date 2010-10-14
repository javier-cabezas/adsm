#include <strings.h>

#include "core/IOBuffer.h"
#include "core/Process.h"
#include "protocol/Lazy.h"

#include "Manager.h"
#include "Map.h"
#include "Object.h"

namespace gmac { namespace memory {

Manager::~Manager()
{
    trace("Memory manager finishes");
    delete protocol_;
}

Manager::Manager()
{
    trace("Memory manager starts");
    // Create protocol
    if(strcasecmp(paramProtocol, "Rolling") == 0) {
        protocol_ = new protocol::Lazy(paramRollSize);
    }
    else if(strcasecmp(paramProtocol, "Lazy") == 0) {
        protocol_ = new protocol::Lazy(-1);
    }
    else {
        Fatal("Memory Coherence Protocol not defined");
    }
}

gmacError_t Manager::alloc(void ** addr, size_t size)
{
    Mode &mode = Mode::current();
    // For integrated devices we want to use Centralized objects to avoid memory transfers
    if (mode.integrated()) return globalAlloc(addr, size, GMAC_GLOBAL_MALLOC_CENTRALIZED);

    gmacError_t ret;
    // Create new shared object
    Object *object = protocol_->createObject(size);
    *addr = object->addr();
    if(*addr == NULL) {
        delete object;
        return gmacErrorMemoryAllocation;
    }

    // Insert object into memory maps
    mode.addObject(*object);

    return gmacSuccess;
}

#ifndef USE_MMAP
gmacError_t Manager::globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint)
{
    gmac::Process &proc = gmac::Process::getInstance();
    Mode &mode = gmac::Mode::current();

    if (proc.allIntegrated()) hint = GMAC_GLOBAL_MALLOC_CENTRALIZED;
    gmacError_t ret;
    if(hint == GMAC_GLOBAL_MALLOC_REPLICATED) {
        Object *object = protocol_->createReplicatedObject(size);
        *addr = object->addr();
        if(*addr == NULL) {
            delete object;
            return gmacErrorMemoryAllocation;
        }

        mode.addReplicatedObject(*object);
    }
    else if (GMAC_GLOBAL_MALLOC_CENTRALIZED) {
        Object *object = protocol_->createCentralizedObject(size);
        *addr = object->addr();
        if(*addr == NULL) {
            delete object;
            return gmacErrorMemoryAllocation;
        }

        mode.addCentralizedObject(*object);
    } else {
        return gmacErrorInvalidValue;
    }

    return gmacSuccess;
}


#endif

gmacError_t Manager::free(void * addr)
{
    gmacError_t ret = gmacSuccess;
    Mode &mode = Mode::current();
    Object *object = mode.getObjectWrite(addr);
    if(object != NULL)  {
        mode.removeObject(*object);
        mode.putObject(*object);
        object->fini();
        delete object;
    }
    else ret = gmacErrorInvalidValue;
    return ret;
}

gmacError_t Manager::acquire()
{
    gmacError_t ret = gmacSuccess;
    Mode &mode = Mode::current();
    if (mode.releasedObjects() == false) { return gmacSuccess; }

    const Map &map = mode.objects();
    map.lockRead();
    Map::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        Object &object = *i->second;
        ret = protocol_->acquire(object);
        if(ret != gmacSuccess) return ret;
    }
    map.unlock();
    mode.releaseObjects();

    return ret;
}

gmacError_t Manager::release()
{
#ifdef USE_VM
    checkBitmapToDevice();
#endif
    Mode &mode = Mode::current();
    trace("Releasing Objects");
    gmacError_t ret = gmacSuccess;
    protocol_->release();

    mode.releaseObjects();
#if 0
    Mode * mode = Mode::current();
    const Map &map = mode->objects();
    map.lockRead();
    Map::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        Object &object = *i->second;
        ret = protocol_->release(object);
        if(ret != gmacSuccess) return ret;
    }
    const ObjectMap &shared = proc->shared();
    ObjectMap::const_iterator j;
    for(j = shared.begin(); j != shared.end(); j++) {
        Object &object = *j->second;
        ret = protocol_->toDevice(object);
        if(ret != gmacSuccess) return ret;
    }
    map.unlock();
#endif
    return ret;
}

#if 0
gmacError_t Manager::invalidate()
{
    abort();
    gmacError_t ret = gmacSuccess;
    const Map &map = Mode::current()->objects();
    Map::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        Object &object = *i->second;
        ret = protocol_->invalidate(object);
        if(ret != gmacSuccess) return ret;
    }
    return ret;
}
#endif

gmacError_t Manager::toIOBuffer(IOBuffer &buffer, const void *addr, size_t count)
{
    assertion(count <= buffer.size());
    gmac::Process &proc = gmac::Process::getInstance();
    gmacError_t ret = gmacSuccess;
    const uint8_t *ptr = (const uint8_t *)addr;
    unsigned off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        Mode * mode = proc.owner(ptr + off);
        if (mode == NULL) return gmacErrorInvalidValue;
        const Object *obj = mode->getObjectRead(ptr + off);
        if (!obj) return gmacErrorInvalidValue;
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (ptr + off);
        size_t c = objCount <= buffer.size() - off? objCount: buffer.size() - off;
        unsigned objOff = ptr - obj->addr();
        // Handle objects with no memory in the accelerator
        if (!obj->isInAccelerator()) {
            ::memcpy(buffer.addr() + off, ptr + off, c);
        } else { // Handle objects with memory in the accelerator
            ret = protocol_->toIOBuffer(buffer, off, *obj, objOff, c);
            if(ret != gmacSuccess) return ret;
        }
        mode->putObject(*obj);
        off += objCount;
        trace("Copying from obj %p: %zd of %zd", obj->addr(), c, count);
    } while(ptr + off < ptr + count);
    return ret;
}

gmacError_t Manager::fromIOBuffer(void * addr, IOBuffer &buffer, size_t count)
{
    assertion(count <= buffer.size());
    gmac::Process &proc = gmac::Process::getInstance();
    gmacError_t ret = gmacSuccess;
    uint8_t *ptr = (uint8_t *)addr;
    unsigned off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        Mode *mode = proc.owner(ptr + off);
        if (mode == NULL) return gmacErrorInvalidValue;
        const Object *obj = mode->getObjectRead(ptr + off);
        if (!obj) return gmacErrorInvalidValue;
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (ptr + off);
        size_t c = objCount <= buffer.size() - off? objCount: buffer.size() - off;
        unsigned objOff = ptr - obj->addr();
        // Handle objects with no memory in the accelerator
        if (!obj->isInAccelerator()) {
            ::memcpy(ptr + off, buffer.addr() + off, c);
        } else { // Handle objects with memory in the accelerator
            ret = protocol_->fromIOBuffer(*obj, objOff, buffer, off, c);
            if(ret != gmacSuccess) return ret;
        }
        mode->putObject(*obj);
        off += objCount;
        trace("Copying to obj %p: %zd of %zd", obj->addr(), c, count);
    } while(ptr + off < ptr + count);
    return ret;
}

#ifdef USE_VM
void Manager::checkBitmapToHost()
{
    Mode *mode = gmac::Mode::current();
    vm::Bitmap &bitmap = mode->dirtyBitmap();
    if (!bitmap.synced()) {
        bitmap.syncHost();

        const Map &map = mode->objects();
        map.lockRead();
        Map::const_iterator i;
        for(i = map.begin(); i != map.end(); i++) {
            Object &object = *i->second;
            gmacError_t ret = protocol_->acquireWithBitmap(object);
            assertion(ret == gmacSuccess);
        }
        map.unlock();
    }
}

void Manager::checkBitmapToDevice()
{
    Mode &mode = gmac::Mode::current();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    if (!bitmap.clean()) {
        bitmap.syncDevice();
    }
}
#endif

bool Manager::read(void *addr)
{
    Mode &mode = gmac::Mode::current();
#ifdef USE_VM
    checkBitmapToHost();
#endif
    bool ret = true;
    const Object *obj = mode.getObjectRead(addr);
    if(obj == NULL) return false;
    trace("Read access for object %p", obj->addr());
    gmacError_t err;
    assertion((err = protocol_->signalRead(*obj, addr)) == gmacSuccess);
    mode.putObject(*obj);
    return ret;
}

bool Manager::write(void *addr)
{
    Mode &mode = gmac::Mode::current();
#ifdef USE_VM
    checkBitmapToHost();
#endif
    bool ret = true;
    const Object *obj = mode.getObjectRead(addr);
    if(obj == NULL) return false;
    trace("Write access for object %p", obj->addr());
    if(protocol_->signalWrite(*obj, addr) != gmacSuccess) ret = false;
    mode.putObject(*obj);
    return ret;
}

#ifndef USE_MMAP
bool Manager::requireUpdate(Block &block)
{
    return protocol_->requireUpdate(block);
}
#endif

gmacError_t Manager::memcpy(void * dst, const void * src, size_t n)
{
    gmac::Process &proc = gmac::Process::getInstance();
    gmac::Mode *dstMode = proc.owner(dst);
    gmac::Mode *srcMode = proc.owner(src);

	if (dstMode == NULL && srcMode == NULL) {
        ::memcpy(dst, src, n);
        return gmacSuccess;
    }

    const Object *dstObj = NULL;
    const Object *srcObj = NULL;
	if (dstMode != NULL) {
        dstObj = dstMode->getObjectRead(dst);
        assertion(dstObj != NULL);
    }
    if (srcMode != NULL) {
        srcObj = srcMode->getObjectRead(src);
        assertion(srcObj != NULL);
    }

    gmacError_t err = gmacSuccess;
    if (dstMode == NULL) {	    // From device
		err = protocol_->toPointer(dst, *srcObj, (uint8_t *)src - srcObj->addr(), n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
	}
    else if(srcMode == NULL) {   // To device
		err = protocol_->fromPointer(*dstObj, (uint8_t *)dst - dstObj->addr(), src, n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
    }
    else {
        if (!srcObj->isInAccelerator() && !dstObj->isInAccelerator()) {
            ::memcpy(dst, src, n);
        } else if (srcObj->isInAccelerator() && !dstObj->isInAccelerator()) {
            err = protocol_->toPointer(dst, *srcObj, (uint8_t *)src - srcObj->addr(), n);
            gmac::util::Logger::ASSERTION(err == gmacSuccess);
        } else if (!srcObj->isInAccelerator() && dstObj->isInAccelerator()) {
            err = protocol_->fromPointer(*dstObj, (uint8_t *)dst - dstObj->addr(), src, n);
            gmac::util::Logger::ASSERTION(err == gmacSuccess);
        } else {
            err = protocol_->copy(*dstObj, (uint8_t *)dst - dstObj->addr(),
                                  *srcObj, (uint8_t *)src - srcObj->addr(), n);
            gmac::util::Logger::ASSERTION(err == gmacSuccess);
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
        gmac::Mode *mode = gmac::Mode::current();
        gmac::util::Logger::ASSERTION(mode != NULL);

        manager->release((void *)src, n);
        manager->invalidate(dst, n);

        off_t off = 0;
        gmac::IOBuffer *buffer = gmac::Mode::current()->getIOBuffer();

        size_t left = n;
        while (left != 0) {
            size_t bytes = left < buffer->size() ? left : buffer->size();
            err = srcMode->bufferToHost(buffer, proc->translate((char *)src + off), bytes);
            gmac::util::Logger::ASSERTION(err == gmacSuccess);

            err = dstMode->bufferToDevice(buffer, proc->translate((char *)dst + off), bytes);
            gmac::util::Logger::ASSERTION(err == gmacSuccess);

            left -= bytes;
            off  += bytes;
        }

	}
#endif
    return err;
}

gmacError_t Manager::memset(void *s, int c, size_t n)
{
    gmac::Process &proc = gmac::Process::getInstance();
    gmac::Mode *mode = proc.owner(s);
	if (mode == NULL) {
        ::memset(s, c, n);
        return gmacSuccess;
    }

    const Object * obj = mode->getObjectRead(s);
    gmacError_t ret;
    ret = protocol_->memset(*obj, (uint8_t *)s - obj->addr(), c, n);
    mode->putObject(*obj);
    return ret;
}

gmacError_t Manager::moveTo(void * addr, Mode &mode)
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

}}
