#include <algorithm>

#include "Lazy.h"

#include "core/IOBuffer.h"

#include "config/config.h"

#include "memory/StateObject.h"
#include "memory/SharedObject.h"
#include "memory/ReplicatedObject.h"
#include "memory/Memory.h"

#include "trace/Function.h"

#include "memory/Block.h"

#if defined(__GNUC__)
#define MIN std::min
#elif defined(_MSC_VER)
#define MIN min
#endif

namespace gmac { namespace memory { namespace protocol { namespace __impl {

List Lazy::GlobalCache_;

bool List::empty() const
{
    bool ret = std::list<Entry>::empty();
    return ret;
}

size_t List::size() const
{
    size_t ret = std::list<Entry>::size();
    return ret;
}

void List::purge(const StateObject<Lazy::State> &obj)
{
    iterator i;
    for(i = begin(); i != end(); ) {
        if(&(i->object) == &obj) i = erase(i);
        else i++;
    }
}

void List::push(const StateObject<Lazy::State> &obj,
        SystemBlock<Lazy::State> *block)
{
    push_back(Entry(obj, block));
}

Entry List::pop() 
{
    Entry ret = front();
    pop_front();
    return ret;
}


Lazy::Lazy(unsigned limit)
    : util::RWLock("Lazy"), _maxListSize(limit)
{
}

Lazy::~Lazy()
{
    lockWrite();
    for(iterator i = begin(); i != end(); i++)
        delete i->second;
    unlock();
}

memory::Object *Lazy::createSharedObject(size_t size, void *cpuPtr, GmacProtection prot)
{
    Object *ret = new SharedObject<Lazy::State>(size, cpuPtr, ReadOnly);
    if (ret != NULL) {
        ret->init();
        if (cpuPtr != NULL) {
            Memory::protect(ret->addr(), ret->size(), GMAC_PROT_READWRITE);
            const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(*ret);
            const StateObject<State>::SystemMap &map = object.blocks();
            StateObject<State>::SystemMap::const_iterator i;
            for(i = map.begin(); i != map.end(); i++) {
                SystemBlock<State> *block = i->second;
                block->lock();
                // TODO helgrind error lock order block dirty_list
                block->state(Dirty);
                gmacError_t err = addDirty(object, *block);
                block->unlock();
            }
        }
    }
    return ret;
}

void Lazy::deleteObject(const Object &obj)
{
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    if (object.isLocal()) {
        lockWrite();
        iterator i;
        for(i = begin(); i != end(); i++) i->second->purge(object);
        unlock();
    } else {
        GlobalCache_.lockWrite();
        GlobalCache_.purge(object);
        GlobalCache_.unlock();
    }
}

#ifndef USE_MMAP
memory::Object *Lazy::createReplicatedObject(size_t size)
{
    Object *ret = new ReplicatedObject<Lazy::State>(size, ReadOnly);
    ret->init();
    return ret;
}
#endif

gmacError_t Lazy::acquire(const Object &obj)
{
    gmacError_t ret = gmacSuccess;
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();
        switch(block->state()) {
            case Invalid:
            case ReadOnly:
				if(Memory::protect(block->addr(), block->size(), GMAC_PROT_NONE) < 0)
                    Fatal("Unable to set memory permissions");
                block->state(Invalid);
                break;
            case Dirty:
                Fatal("Block in incongruent state in acquire: %p", block->addr());
                break;
        }
        block->unlock();
    }
    return ret;
}

#ifdef USE_VM
gmacError_t Lazy::acquireWithBitmap(const Object &obj)
{
    Mode &mode = gmac::Mode::current();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    gmacError_t ret = gmacSuccess;
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();
        if (bitmap.check(obj.device(block->addr()))) {
            if(Memory::protect(block->addr(), block->size(), GMAC_PROT_NONE) < 0)
                Fatal("Unable to set memory permissions");
            block->state(Invalid);
        } else {
            if(Memory::protect(block->addr(), block->size(), GMAC_PROT_READ) < 0)
                Fatal("Unable to set memory permissions");
            block->state(ReadOnly);
        }
        block->unlock();
    }
    return ret;
}
#endif

gmacError_t Lazy::release(const StateObject<State> &object, SystemBlock<State> &block)
{
    trace("Releasing block %p", block.addr());
    gmacError_t ret = gmacSuccess;
    block.lock();
    switch(block.state()) {
        case Dirty:
            ret = object.toAccelerator(block);
            if(ret != gmacSuccess) goto exit_func;
			if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                    Fatal("Unable to set memory permissions");
            block.state(ReadOnly);
            break;

        case Invalid:
        case ReadOnly:
            break;
    }
exit_func:
    block.unlock();
    return ret;
}


gmacError_t Lazy::release()
{
    // Release global dirty blocks
    GlobalCache_.lockWrite();
    while(GlobalCache_.empty() == false) {
        Entry e = GlobalCache_.pop();
        gmacError_t ret = release(e.object, *e.block);
        if(ret != gmacSuccess) {
            GlobalCache_.unlock();
            return ret;
        }
    }
    GlobalCache_.unlock();

    // Get the dirty list for the current mode
    lockRead();
    iterator i = find(&gmac::Mode::current());
    if(i == end()) {
        unlock();
        return gmacSuccess;
    }
    List &list = *i->second;
    unlock();

    list.lockWrite();
    trace("Cache contains %zd elements\n", list.size());
    // Release dirty blocks
    while(list.empty() == false) {
        Entry e = list.pop();
        gmacError_t ret = release(e.object, *e.block);
        if(ret != gmacSuccess) {
            list.unlock();
            return ret;
        }
    }
    list.unlock();

    return gmacSuccess;
}

gmacError_t Lazy::toHost(const Object &obj)
{
    gmacError_t ret = gmacSuccess;
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> &block = *i->second;
        block.lock();
        switch(block.state()) {
            case Invalid:
				if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READWRITE) < 0)
                    Fatal("Unable to set memory permissions");
                ret = object.toHost(block);
                if(ret != gmacSuccess) { block.unlock(); return ret; }
                block.state(ReadOnly);
                break;

            case Dirty:
            case ReadOnly:
                break;
        }
        block.unlock();
    }
    return ret;
}

gmacError_t Lazy::toDevice(const Object &obj)
{
    gmacError_t ret = gmacSuccess;
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> &block = *i->second;
        block.lock();
        switch(block.state()) {
            case Dirty:
                ret = object.toAccelerator(block);
                if(ret != gmacSuccess) {
                    block.unlock();
                    return ret;
                }
				if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                    Fatal("Unable to set memory permissions");
                block.state(ReadOnly);
            break;

        case Invalid:
        case ReadOnly:
            break;
        }
        block.unlock();
    }
    return ret;
}

static size_t blockRemainder(const uint8_t * blockAddr, size_t blockSize, const uint8_t * ptr, size_t n)
{
    if (ptr >= blockAddr && ptr + n <  blockAddr + blockSize) {
        return unsigned(n);
    } else if (ptr <  blockAddr && ptr + n < blockAddr + blockSize) {
        return unsigned(ptr + n - blockAddr);
    } else if (ptr <  blockAddr && ptr + n >= blockAddr + blockSize) {
        return unsigned(blockSize);
    } else { // if (ptr >= blockAddr && ptr + n >= blockAddr + blockSize) {
        return unsigned(blockAddr + blockSize - ptr);
    }
}

gmacError_t
Lazy::toIOBuffer(IOBuffer &buffer, unsigned bufferOff, const Object &obj, unsigned objectOff, size_t count)
{
    trace::Function::start("Lazy", "toIOBuffer");
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * addr = object.addr() + objectOff;
    i = object.getBlockIterator(addr);
    assertion(i != map.end());

    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        size_t bytes = blockRemainder(block.addr(), block.size(), addr, count);
        unsigned blockOff = unsigned(addr + off - block.addr());

        switch(block.state()) {
            case Dirty:
            case ReadOnly:
                ::memcpy(buffer.addr() + bufferOff + off, addr + off, bytes);
                break;

            case Invalid:
                ret = object.toHostBuffer(block, blockOff, buffer, bufferOff + off, bytes);
                if(ret != gmacSuccess) {
                    block.unlock();
                    goto exit_func;
                }
                break;
        }
        off += unsigned(bytes);
        block.unlock();
        i++;
    } while (off < count);

exit_func:
    trace::Function::end("Lazy");
    return ret;
}

gmacError_t
Lazy::fromIOBuffer(const Object &obj, unsigned objectOff, IOBuffer &buffer, unsigned bufferOff, size_t count)
{
    trace::Function::start("Lazy", "fromIOBuffer");
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * addr = object.addr() + objectOff;
    i = object.getBlockIterator(addr);
    assertion(i != map.end());

    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        size_t bytes = blockRemainder(block.addr(), block.size(), addr, count);
        off_t blockOff = (off_t)(addr + off - block.addr());

        switch(block.state()) {
            case Dirty:
                ::memcpy(addr + off, buffer.addr() + bufferOff + off, bytes);
                break;

            case ReadOnly:
				if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READWRITE) < 0)
                    Fatal("Unable to set memory permissions");
                ::memcpy(addr + off,  buffer.addr() + off, bytes);
				if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                    Fatal("Unable to set memory permissions");
                ret = object.toAccelerator(block);
                break;

            case Invalid:
                ret = object.toAcceleratorFromBuffer(block, blockOff, buffer, bufferOff + off, bytes);
                if(ret != gmacSuccess) {
                    block.unlock();
                    goto exit_func;
                }
                break;
        }           
        off += (off_t)bytes;
        block.unlock();
        i++;
    } while (off < count);

exit_func:
    trace::Function::end("Lazy");
    return ret;
}

gmacError_t
Lazy::toPointer(void *_dst, const Object &objSrc, unsigned objectOff, size_t count)
{
    trace::Function::start("Lazy", "toPointer");
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(objSrc);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * dst = (uint8_t *)_dst;
    const uint8_t * src = (const uint8_t *) object.addr() + objectOff;
    i = object.getBlockIterator(src);
    assertion(i != map.end());

    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        size_t bytes = blockRemainder(block.addr(), block.size(), src, count);
        unsigned blockOff = unsigned(src + off - block.addr());

        switch(block.state()) {
            case Dirty:
            case ReadOnly:
                ::memcpy(dst + off, src + off, bytes);
                break;

            case Invalid:
                ret = object.toHostPointer(block, blockOff, dst + off, bytes);
                if(ret != gmacSuccess) {
                    block.unlock();
                    goto exit_func;
                }

                break;
        }
        off += unsigned(bytes);
        block.unlock();
        i++;
    } while (off < count);

exit_func:
    trace::Function::end("Lazy");
    return ret;
}

gmacError_t
Lazy::fromPointer(const Object &objDst, unsigned objectOff, const void *_src, size_t count)
{
    trace::Function::start("Lazy", "fromPointer");
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(objDst);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * dst = object.addr() + objectOff;
    const uint8_t * src = (const uint8_t *) _src;
    i = object.getBlockIterator(dst);
    assertion(i != map.end());

    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        size_t bytes = blockRemainder(block.addr(), block.size(), dst, count);
        unsigned blockOff = unsigned(dst + off - block.addr());

        switch(block.state()) {
            case Dirty:
                ::memcpy(dst + off, src + off, bytes);
                break;

            case ReadOnly:
				if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READWRITE) < 0)
                    Fatal("Unable to set memory permissions");
                ::memcpy(dst + off, src + off, bytes);
				if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                    Fatal("Unable to set memory permissions");
                ret = object.toAccelerator(block);
                break;

            case Invalid:
                ret = object.toAcceleratorFromPointer(block, blockOff, src + off, bytes);
                if(ret != gmacSuccess) {
                    block.unlock();
                    goto exit_func;
               }
                break;
        }           
        off += bytes;
        block.unlock();
        i++;
    } while (off < count);

exit_func:
    trace::Function::end("Lazy");
    return ret;
} 

inline
gmacError_t
Lazy::copyHostToDirty(const StateObject<State> &/*objectDst*/, Block &blockDst, unsigned blockOffDst,
                      const StateObject<State> &/*objectSrc*/, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // Host memory to host memory
    ::memcpy(blockDst.addr() + blockOffDst,
             blockSrc.addr() + blockOffSrc, count);
    return gmacSuccess;
}

gmacError_t
Lazy::copyHostToReadOnly(const StateObject<State> &/*objectDst*/, Block &blockDst, unsigned blockOffDst,
                         const StateObject<State> &/*objectSrc*/, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // Host memory to host memory AND accelerator memory
	uint8_t *tmp = (uint8_t *)Memory::shadow(blockDst.addr(), blockDst.size());
    if (tmp == NULL) {
        return gmacErrorInvalidValue;
    }
    ::memcpy(tmp + blockOffDst, blockSrc.addr() + blockOffSrc, count);
	Memory::unshadow(tmp, blockDst.size());
    return gmacSuccess;
}

inline
gmacError_t
Lazy::copyHostToInvalid(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                        const StateObject<State> &/*objectSrc*/, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // Host memory to accelerator memory
    return objectDst.toAcceleratorFromPointer(blockDst, blockOffDst, blockSrc.addr() + blockOffSrc, count);
}


gmacError_t
Lazy::copyAcceleratorToDirty(const StateObject<State> &/*objectDst*/, Block &/*blockDst*/, unsigned /*blockOffDst*/,
                             const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // Accelerator memory to host memory
    return objectSrc.toHostPointer(blockSrc, blockOffSrc, blockSrc.addr() + blockOffSrc, count);
}

gmacError_t
Lazy::copyAcceleratorToReadOnly(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                                const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
	uint8_t *tmp = (uint8_t *)Memory::shadow(blockDst.addr(), blockDst.size());
    if (tmp == NULL) {
        return gmacErrorInvalidValue;
    }

    gmacError_t ret = objectSrc.toHostPointer(blockSrc, blockOffSrc, tmp + blockOffDst, count);
    if (ret != gmacSuccess) {
        return gmacErrorInvalidValue;
    }

    Memory::unshadow(tmp, blockDst.size());

    ret = objectDst.toAccelerator(blockDst, blockOffDst, count);
    return ret;
}

gmacError_t
Lazy::copyAcceleratorToInvalid(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                               const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    IOBuffer *buffer = Mode::current().createIOBuffer(count); 
    if (!buffer) {
        void *tmp = Memory::map(NULL, count, GMAC_PROT_READWRITE);
        CFatal(tmp != NULL, "Unable to set memory permissions");
        gmacError_t ret = objectSrc.toHostPointer(blockSrc, blockOffSrc, tmp, count);
        if (ret != gmacSuccess) return ret;
        ret = objectDst.toAcceleratorFromPointer(blockDst, blockOffDst, tmp, count);
        if (ret != gmacSuccess) return ret;
        Memory::unmap(tmp, count);
    } else {
        gmacError_t ret = objectSrc.toHostBuffer(blockSrc, blockOffSrc, *buffer, 0, count);
        if (ret != gmacSuccess) return ret;
        ret = buffer->wait();
        if (ret != gmacSuccess) return ret;
        ret = objectDst.toAcceleratorFromBuffer(blockDst, blockOffDst, *buffer, 0, count);
        if (ret != gmacSuccess) return ret;
        ret = buffer->wait();
        if (ret != gmacSuccess) return ret;
        Mode::current().destroyIOBuffer(buffer);
    }
    return gmacSuccess;
}

gmacError_t
Lazy::copy(const Object &objDst, unsigned offDst, const Object &objSrc, unsigned offSrc, size_t count)
{
    trace::Function::start("Lazy", "copy");
    gmacError_t ret = gmacSuccess;
    
    const StateObject<State> &objectDst = dynamic_cast<const StateObject<State> &>(objDst);
    const StateObject<State> &objectSrc = dynamic_cast<const StateObject<State> &>(objSrc);

    const StateObject<State>::SystemMap &mapDst = objectDst.blocks();
    const StateObject<State>::SystemMap &mapSrc = objectSrc.blocks();
    unsigned off = 0;

    uint8_t * dst = objectDst.addr() + offDst;
    uint8_t * src = objectSrc.addr() + offSrc;

    StateObject<State>::SystemMap::const_iterator iDst = objectDst.getBlockIterator(dst);
    StateObject<State>::SystemMap::const_iterator iSrc = objectSrc.getBlockIterator(src);
    assertion(iDst != mapDst.end());
    assertion(iSrc != mapSrc.end());
    do {
        SystemBlock<State> &blockDst = *iDst->second;
        SystemBlock<State> &blockSrc = *iSrc->second;
        blockDst.lock();
        blockSrc.lock();

        size_t bytesDst = blockRemainder(blockDst.addr(), blockDst.size(), dst, count);
        size_t bytesSrc = blockRemainder(blockSrc.addr(), blockSrc.size(), src, count);

        size_t bytes = bytesSrc != bytesDst? MIN(bytesSrc, bytesDst): bytesSrc;

        unsigned blockOffDst = unsigned(dst + off - blockDst.addr());
        unsigned blockOffSrc = unsigned(src + off - blockSrc.addr());

        switch(blockSrc.state()) {
            // Source location present in host memory
            case Dirty:
            case ReadOnly:
                switch(blockDst.state()) {
                    case Dirty:
                        ret = copyHostToDirty(objectDst, blockDst, blockOffDst,
                                              objectSrc, blockSrc, blockOffSrc, bytes);
                        break;
                    case ReadOnly:
                        ret = copyHostToReadOnly(objectDst, blockDst, blockOffDst,
                                                 objectSrc, blockSrc, blockOffSrc, bytes);
                        break;
                    case Invalid:
                        ret = copyHostToInvalid(objectDst, blockDst, blockOffDst,
                                                objectSrc, blockSrc, blockOffSrc, bytes);
                        break;
                }
                break;

            // Source location in accelerator memory
            case Invalid:
                switch(blockDst.state()) {
                    case Dirty:
                        ret = copyHostToDirty(objectDst, blockDst, blockOffDst,
                                              objectSrc, blockSrc, blockOffSrc, bytes);
                        break;
                    // Accelerator memory to host memory AND accelerator memory
                    case ReadOnly:
                        ret = copyHostToReadOnly(objectDst, blockDst, blockOffDst,
                                                 objectSrc, blockSrc, blockOffSrc, bytes);
                        break;
                    // Host memory to accelerator memory
                    case Invalid:
                        ret = copyHostToInvalid(objectDst, blockDst, blockOffDst,
                                                objectSrc, blockSrc, blockOffSrc, bytes);
                        break;
                }
                break;
        }           
        blockDst.unlock();
        blockSrc.unlock();
        if(ret != gmacSuccess) {
            goto exit_func;
        }
        off += bytes;
        if (blockDst.addr() + bytesDst == blockDst.end()) {
            iDst++;
        }
        if (blockSrc.addr() + bytesSrc == blockSrc.end()) {
            iSrc++;
        }
    } while (off < count);

exit_func:
    trace::Function::end("Lazy");
    return ret;
}

gmacError_t
Lazy::memset(const Object &obj, unsigned objectOff, int c, size_t count)
{
    trace::Function::start("Lazy", "memset");
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * s = object.addr() + objectOff;
    i = object.getBlockIterator(s);
    assert(i != map.end());
    uint8_t *tmp = NULL;
    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        unsigned bytes = blockRemainder(block.addr(), block.size(), s, count);

        switch(block.state()) {
            case Dirty:
                ::memset(s + off, c, bytes);
                break;

            case ReadOnly:
                ret = object.memsetAccelerator(block, unsigned(s + off - block.addr()), c, bytes);
				tmp = (uint8_t *)Memory::shadow(block.addr(), block.size());
				if(tmp == NULL)
                    Fatal("Unable to create shadow memory copy");
                ::memset(tmp + objectOff + off, c, bytes);
				Memory::unshadow(tmp, block.size());
                ret = object.toAccelerator(block);
                break;

            case Invalid:
                ret = object.memsetAccelerator(block, unsigned(s + off - block.addr()), c, bytes);
                if(ret != gmacSuccess) {
                    block.unlock();
                    goto exit_func;
                }
                break;
        }           
        block.unlock();
        off += (off_t)bytes;
        i++;
    } while (off < count);

exit_func:
    trace::Function::end("Lazy");
    return ret;
}

gmacError_t
Lazy::moveTo(Object &obj, Mode &mode)
{
    trace::Function::start("Lazy", "moveTo");
    gmacError_t ret = gmacSuccess;
    ret = toHost(obj);
    if (ret == gmacSuccess) {
        StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
        ret = object.realloc(mode);
    }
    trace::Function::end("Lazy");
    return ret;
}

gmacError_t Lazy::signalRead(const Object &obj, void *addr)
{
    trace::Function::start("Lazy", "signalRead");
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    block->lock();
    gmacError_t ret = gmacSuccess;

    if (block->state() != Invalid) {
        // Somebody already fixed it
        goto exit_func;
    }
    void * tmp;
#ifdef USE_VM
    Mode &mode = Mode::current();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    if (bitmap.checkAndClear(obj.device(block->addr()))) {
#endif
		tmp = Memory::shadow(block->addr(), block->size());
        if (tmp == NULL) {
            ret = gmacErrorInvalidValue;
            goto exit_func;
        }

        ret = object.toHostPointer(*block, 0, tmp, block->size());
        Memory::unshadow(tmp, block->size());
        if(ret != gmacSuccess) {
            goto exit_func;
        }
            
        Memory::protect(block->addr(), block->size(), GMAC_PROT_READ);
#ifdef USE_VM
    }
#endif
    block->state(ReadOnly);
exit_func:
    block->unlock();
    trace::Function::end("Lazy");
    return ret;
}

gmacError_t Lazy::signalWrite(const Object &obj, void *addr)
{
    trace::Function::start("Lazy", "signalWrite");
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    gmacError_t ret = gmacSuccess;
    block->lock();
    switch (block->state()) {
        case Dirty:
            // Somebody already fixed it
            goto exit_func;
        case Invalid:
            void * tmp;
#ifdef USE_VM
            vm::Bitmap &bitmap = mode.dirtyBitmap();
            if (bitmap.checkAndClear(obj.device(block->addr()))) {
#endif
				tmp = Memory::shadow(block->addr(), block->size());
                if (tmp == NULL) {
                    ret = gmacErrorInvalidValue;
                    goto exit_func;
                }

                ret = object.toHostPointer(*block, 0, tmp, block->size());
                Memory::unshadow(tmp, block->size());
                if(ret != gmacSuccess) {
                    goto exit_func;
                }
#ifdef USE_VM
            }
#endif
        case ReadOnly:
			Memory::protect(block->addr(), block->size(), GMAC_PROT_READWRITE);
            break;
    }
    block->state(Dirty);
    trace("Setting block %p to dirty state", block->addr());
    // TODO helgrind error lock order block dirty_list
    ret = addDirty(object, *block);
exit_func:
    block->unlock();
    trace::Function::end("Lazy");
    return ret;
}

gmacError_t
Lazy::addDirty(const StateObject<State> &object, SystemBlock<State> &block, bool checkOverflow)
{
    Mode &mode = object.owner();

    // Handle local mode allocations
    if (object.isLocal()) {
        lockWrite();
        iterator i = find(&mode);
        if(i == end()) {
            i = insert(value_type(&mode, new List())).first;
        }
        assertion(i != end());
        List &list = *i->second;
        unlock();

        list.lockWrite();
        list.push(object, &block);

        trace("Adding %p to dirty list %p (%zd)", block.addr(), &list, list.size());

        if (checkOverflow) {
            // Release dirty blocks
            while(list.size() > _maxListSize) {
                Entry e = list.pop();
                trace("Eagerly transferring %p", e.block->addr());
                gmacError_t ret = release(e.object, *e.block);
                if(ret != gmacSuccess) {
                    list.unlock();
                    return ret;
                }
            }
        }
        list.unlock();
    } else { // Handle global allocations
        GlobalCache_.lockWrite();
        GlobalCache_.push(object, &block);

        if (checkOverflow) {
            // Release dirty blocks
            while(GlobalCache_.size() > _maxListSize) {
                Entry e = GlobalCache_.pop();
                // TODO Perform this transfer out of the lock
                gmacError_t ret = release(e.object, *e.block);
                if(ret != gmacSuccess) {
                    GlobalCache_.unlock();
                    return ret;
                }
            }
        }
        GlobalCache_.unlock();
    }
    return gmacSuccess;
}

#ifndef USE_MMAP
bool Lazy::requireUpdate(Block &b)
{
    bool ret = true;
    b.lock();
    SystemBlock<State> &block = dynamic_cast<SystemBlock<State> &>(b);
    switch(block.state()) {
        case Dirty:
            ret = false; break;
        case Invalid:
        case ReadOnly:
            ret = true; break;
    }
    b.unlock();
    return ret;
}

#endif

gmacError_t
Lazy::removeMode(Mode &mode)
{
    lockWrite();
    iterator i = find(&mode);
    if(i != end()) {
        delete i->second;
        erase(i);
    }
    unlock();
    return gmacSuccess;
}

}}}}
