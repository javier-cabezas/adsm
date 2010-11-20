#include "Lazy.h"

#include "core/IOBuffer.h"

#include "config/config.h"

#include "memory/Memory.h"
#include "memory/SharedObject.h"
#include "memory/DistributedObject.h"

#include "memory/StateBlock.h"

#include "trace/Tracer.h"


#if defined(__GNUC__)
#define MIN std::min
#elif defined(_MSC_VER)
#define MIN min
#endif

namespace __impl { namespace memory { namespace protocol {


Lazy::Lazy(unsigned limit) :
    gmac::util::RWLock("Lazy"),
    limit_(limit)
{
}

Lazy::~Lazy()
{

}

Lazy::State Lazy::state(GmacProtection prot) const
{
	switch(prot) {
		case GMAC_PROT_NONE: 
			return Invalid;
		case GMAC_PROT_READ:
			return ReadOnly;
		case GMAC_PROT_WRITE:
		case GMAC_PROT_READWRITE:
			return Dirty;
	}
	return Dirty;
}

memory::Object *Lazy::createObject(size_t size, void *cpuPtr, GmacProtection prot)
{
	Object *ret = new SharedObject<State>(*this, core::Mode::current(), cpuPtr, 
		size, state(prot));
	if(ret == NULL) return ret;
	if(ret->addr() == NULL) {
		ret->release();
		return NULL;
	}
	Memory::protect(ret->addr(), ret->size(), prot);
	return ret;
}

memory::Object *Lazy::createGlobalObject(size_t size, void *cpuPtr, 
                                         GmacProtection prot, unsigned /*flags*/)
{
	Object *ret = new DistributedObject<State>(*this, core::Mode::current(), cpuPtr,
		size, state(prot));
	if(ret == NULL) return ret;
	if(ret->addr() == NULL) {
		ret->release();
		return NULL;
	}
	Memory::protect(ret->addr(), ret->size(), prot);
	return ret;
}

void Lazy::deleteObject(Object &obj)
{
    // TODO: purge blocks in list
    obj.release();
}

bool Lazy::needUpdate(const Block &b) const
{
    const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
    switch(block.state()) {        
        case Dirty:
        case HostOnly:
            return false;
        case ReadOnly:
        case Invalid:
            return true;
    }
    return false;
}

gmacError_t Lazy::signalRead(Block &b)
{
    trace::EnterCurrentFunction();
	StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    gmacError_t ret = gmacSuccess;

    if(block.state() == HostOnly) {
        WARNING("Signal on HostOnly block - Changing protection and continuing");
        Memory::protect(block.addr(), block.size(), GMAC_PROT_READWRITE);
        goto exit_func;
    }


    if (block.state() != Invalid) {
        goto exit_func; // Somebody already fixed it
    }
#ifdef USE_VM
    gmac::core::Mode &mode = gmac::core::Mode::current();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    if (bitmap.checkAndClear(obj.device(block->addr()))) {
#endif
        ret = block.toHost();
        if(ret != gmacSuccess) goto exit_func;
        Memory::protect(block.addr(), block.size(), GMAC_PROT_READ);
#ifdef USE_VM
    }
#endif
    block.state(ReadOnly);

exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Lazy::signalWrite(Block &b)
{
    trace::EnterCurrentFunction();
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    gmacError_t ret = gmacSuccess;
    switch(block.state()) {
        case Dirty:            
            goto exit_func; // Somebody already fixed it
        case Invalid:          
#ifdef USE_VM
            vm::Bitmap &bitmap = mode.dirtyBitmap();
            if (bitmap.checkAndClear(obj.device(block->addr()))) {
#endif
                ret = block.toHost();
                if(ret != gmacSuccess) goto exit_func;
#ifdef USE_VM
            }
#endif
        case HostOnly:
            WARNING("Signal on HostOnly block - Changing protection and continuing");
        case ReadOnly:
			Memory::protect(block.addr(), block.size(), GMAC_PROT_READWRITE);
            break;
    }
    block.state(Dirty);
    addDirty(block);
    TRACE(LOCAL,"Setting block %p to dirty state", block.addr());
    //ret = addDirty(block);
exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Lazy::acquire(Block &b)
{
    gmacError_t ret = gmacSuccess;
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Invalid:
        case ReadOnly:
			if(Memory::protect(block.addr(), block.size(), GMAC_PROT_NONE) < 0)
                FATAL("Unable to set memory permissions");
            block.state(Invalid);
            break;
        case Dirty:
            FATAL("Block in incongruent state in acquire: %p", block.addr());
            break;
        case HostOnly:
            break;
    }
	return ret;
}

#ifdef USE_VM
gmacError_t Lazy::acquireWithBitmap(const Object &obj)
{
    core::Mode &mode = core::Mode::current();
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
                FATAL("Unable to set memory permissions");
            block->state(Invalid);
        } else {
            if(Memory::protect(block->addr(), block->size(), GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            block->state(ReadOnly);
        }
        block->unlock();
    }
    return ret;
}
#endif

gmacError_t Lazy::release(StateBlock<State> &block)
{
    TRACE(LOCAL,"Releasing block %p", block.addr());
    gmacError_t ret = gmacSuccess;
    switch(block.state()) {
        case Dirty:
            ret = block.toDevice();
            if(ret != gmacSuccess) break;
			if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                    FATAL("Unable to set memory permissions");
            block.state(ReadOnly);
            break;
        case Invalid:
        case ReadOnly:
        case HostOnly:
            break;
    }
    return ret;
}

gmacError_t Lazy::remove(Block &b)
{
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    TRACE(LOCAL,"Releasing block %p", block.addr());
    gmacError_t ret = gmacSuccess;
    switch(block.state()) {
        case HostOnly:
        case Dirty:
        case ReadOnly:
            break;
        case Invalid:
            ret = block.toHost();
            if(ret != gmacSuccess) break;
    }
    if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READWRITE) < 0)
                    FATAL("Unable to set memory permissions");
    block.state(HostOnly);
    dbl_.remove(block);
    return ret;
}

void Lazy::addDirty(Block &block)
{
    dbl_.push(block);
    if(limit_ == unsigned(-1)) return;
    while(dbl_.size() > limit_) {
        Block *b = dbl_.pop();
        release(dynamic_cast<StateBlock<State> &>(*b));
    }
    return;
}

gmacError_t Lazy::release()
{
    while(dbl_.empty() == false) {
        Block *b = dbl_.pop();
        release(dynamic_cast<StateBlock<State> &>(*b));
    }
    return gmacSuccess;
}

gmacError_t Lazy::toHost(Block &b)
{
    gmacError_t ret = gmacSuccess;
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Invalid:
			if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READWRITE) < 0)
                FATAL("Unable to set memory permissions");
            ret = block.toHost();
            if(ret != gmacSuccess) break;
            block.state(ReadOnly);
            break;
        case Dirty:
        case ReadOnly:
        case HostOnly:
            break;
    }
    return ret;
}

gmacError_t Lazy::toDevice(Block &b)
{
    gmacError_t ret = gmacSuccess;
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Dirty:
            ret = block.toDevice();
            if(ret != gmacSuccess) break;
			if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            block.state(ReadOnly);
        break;
    case Invalid:
    case ReadOnly:
    case HostOnly:
        break;
    }
    return ret;
}

gmacError_t Lazy::copyToBuffer(const Block &b, core::IOBuffer &buffer, size_t size,
							   unsigned bufferOffset, unsigned objectOffset) const
{
	gmacError_t ret = gmacSuccess;
	const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
	switch(block.state()) {
		case Invalid:
			ret = block.copyFromDevice(buffer, size, bufferOffset, objectOffset);
			break;
		case ReadOnly:
		case Dirty:
        case HostOnly:
			ret = block.copyFromHost(buffer, size, bufferOffset, objectOffset);
	}
	return ret;
}

gmacError_t Lazy::copyFromBuffer(const Block &b, core::IOBuffer &buffer, size_t size, 
							   unsigned bufferOffset, unsigned objectOffset) const
{
	gmacError_t ret = gmacSuccess;
	const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
	switch(block.state()) {
		case Invalid:
			ret = block.copyToDevice(buffer, size, bufferOffset, objectOffset);
			break;
		case ReadOnly:
			ret = block.copyToDevice(buffer, size, bufferOffset, objectOffset);
			if(ret != gmacSuccess) break;
			ret = block.copyToHost(buffer, size, bufferOffset, objectOffset);
			break;
		case Dirty:			
        case HostOnly:
			ret = block.copyToHost(buffer, size, bufferOffset, objectOffset);
			break;
	}
	return ret;
}



}}}
