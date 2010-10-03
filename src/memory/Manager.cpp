#include "core/Process.h"

#include "Manager.h"
#include "Map.h"
#include "Object.h"

#include "protocol/Lazy.h"

#include <strings.h>

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
    gmacError_t ret;
    // Create new shared object
    Object *object = protocol_->createObject(size);
    *addr = object->addr();
    if(*addr == NULL) {
        delete object;
        return gmacErrorMemoryAllocation;
    }

    // Insert object into memory maps
    Mode::current().addObject(object);

    return gmacSuccess;
}

#ifndef USE_MMAP
gmacError_t Manager::globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint)
{
    Mode &mode = gmac::Mode::current();
    gmacError_t ret;
    if(hint == GMAC_GLOBAL_MALLOC_REPLICATED) {
        Object *object = protocol_->createReplicatedObject(size);
        *addr = object->addr();
        if(*addr == NULL) {
            delete object;
            return gmacErrorMemoryAllocation;
        }

        mode.addReplicatedObject(object);
    }
    else if (GMAC_GLOBAL_MALLOC_CENTRALIZED) {
        Object *object = protocol_->createCentralizedObject(size);
        *addr = object->addr();
        if(*addr == NULL) {
            delete object;
            return gmacErrorMemoryAllocation;
        }

        mode.addCentralizedObject(object);
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
        mode.removeObject(object);
        mode.putObject(*object);
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

gmacError_t Manager::toIOBuffer(IOBuffer &buffer, const void *addr, size_t size)
{
    gmacError_t ret = gmacSuccess;
    const uint8_t *ptr = (const uint8_t *)addr;
    Mode &mode = Mode::current();
    do {
        const Object *obj = mode.getObjectRead(ptr);
        ret = protocol_->toIOBuffer(buffer, *obj, addr, size);
        ptr += obj->size();
        mode.putObject(*obj);
        if(ret != gmacSuccess) return ret;
    } while(ptr < (uint8_t *)addr + size);
    return ret;
}

gmacError_t Manager::fromIOBuffer(void * addr, IOBuffer &buffer, size_t size)
{
    gmac::Process &proc = gmac::Process::current();
    gmacError_t ret = gmacSuccess;
    uint8_t *ptr = (uint8_t *)addr;
    do {
        gmac::Mode &mode = *proc.owner(addr);
        const Object *obj = mode.getObjectRead(ptr);
        ret = protocol_->fromIOBuffer(buffer, *obj, addr, size);
        ptr += obj->size();
        mode.putObject(*obj);
        if(ret != gmacSuccess) return ret;
    } while(ptr < (uint8_t *)addr + size);
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
    assertion(protocol_->read(*obj, addr) == gmacSuccess);
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
    if(protocol_->write(*obj, addr) != gmacSuccess) ret = false;
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
    gmac::Process &proc = gmac::Process::current();
    gmac::Mode *dstMode = proc.owner(dst);
    gmac::Mode *srcMode = proc.owner(src);

	if (dstMode == NULL && srcMode == NULL) {
        memcpy(dst, src, n);
        return gmacSuccess;
    }

    const Object *dstObj = NULL;
    const Object *srcObj = NULL;
	if (dstMode == NULL) {
        dstObj = dstMode->getObjectRead(dst);
    }
    if (srcMode == NULL) {
        srcObj = srcMode->getObjectRead(src);
    }
    gmacError_t err;
    if (dstMode == NULL) {	    // From device
		err = protocol_->toPointer(dst, src, *srcObj, n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
	}
    else if(srcMode == NULL) {   // To device
		err = protocol_->fromPointer(dst, src, *srcObj, n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
    }
    else {
		err = protocol_->copy(dst, src, *dstObj, *srcObj, n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
	}

    if (dstMode == NULL) {
        dstMode->putObject(*dstObj);
    }
    if (srcMode == NULL) {
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
    gmac::Process &proc = gmac::Process::current();
    gmac::Mode *mode = proc.owner(s);
	if (mode == NULL) {
        ::memset(s, c, n);
        return gmacSuccess;
    }

    const Object * obj = mode->getObjectRead(s);
    gmacError_t ret;
    ret = protocol_->memset(*obj, s, c, n);
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
