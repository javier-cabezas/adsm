#include "Lazy.h"

#include "core/IOBuffer.h"

#include "config/config.h"

#include "memory/Memory.h"
#include "memory/StateBlock.h"

#include "trace/Tracer.h"


#if defined(__GNUC__)
#define MIN std::min
#elif defined(_MSC_VER)
#define MIN min
#endif

namespace __impl { namespace memory { namespace protocol {


LazyBase::LazyBase(unsigned limit) :
    gmac::util::Lock("Lazy"),
    limit_(limit)
{
}

LazyBase::~LazyBase()
{

}

LazyBase::State LazyBase::state(GmacProtection prot) const
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


void LazyBase::deleteObject(Object &obj)
{
    obj.release();
}



bool LazyBase::needUpdate(const Block &b) const
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

gmacError_t LazyBase::signalRead(Block &b, hostptr_t addr)
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

    ret = block.toHost();
    if(ret != gmacSuccess) goto exit_func;
    Memory::protect(block.addr(), block.size(), GMAC_PROT_READ);
    block.state(ReadOnly);

exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t LazyBase::signalWrite(Block &b, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    gmacError_t ret = gmacSuccess;

    hostptr_t start  = block.addr();
    size_t count = block.size();

    switch(block.state()) {
        case Dirty:            
#ifdef USE_VM
            if (!block.isSequentialAccess()) {
                start = block.getSubBlockAddr(addr);
                count = block.getSubBlockSize();

                block.setSubBlockPresent(addr);
            } else {
                block.setBlockPresent();
            }
            Memory::protect(start, count, GMAC_PROT_READWRITE);
#endif
            goto exit_func; // Somebody already fixed it
        case Invalid:          
            ret = block.toHost();
            if(ret != gmacSuccess) goto exit_func;
#ifdef USE_VM
            if (!block.isSequentialAccess()) {
                start = block.getSubBlockAddr(addr);
                count = block.getSubBlockSize();

                block.setSubBlockPresent(addr);
            } else {
                block.setBlockPresent();
            }
#endif
			Memory::protect(start, count, GMAC_PROT_READWRITE);
            break;
        case HostOnly:
            WARNING("Signal on HostOnly block - Changing protection and continuing");
        case ReadOnly:
#ifdef USE_VM
            if (!block.isSequentialAccess()) {
                start = block.getSubBlockAddr(addr);
                count = block.getSubBlockSize();

                block.setSubBlockPresent(addr);
            } else {
                block.setBlockPresent();
            }
#endif
			Memory::protect(start, count, GMAC_PROT_READWRITE);
            break;
    }
    block.state(Dirty);
    addDirty(block);
    TRACE(LOCAL,"Setting block %p to dirty state", start);
    //ret = addDirty(block);
exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t LazyBase::acquire(Block &b)
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
gmacError_t LazyBase::acquireWithBitmap(Block &b)
{
    gmacError_t ret = gmacSuccess;
    core::Mode &mode = core::Mode::current();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Invalid:
        case ReadOnly:
            if (bitmap.checkBlock(block.acceleratorAddr(block.addr()))) {
                if(Memory::protect(block.addr(), block.size(), GMAC_PROT_NONE) < 0)
                    FATAL("Unable to set memory permissions");
                block.state(Invalid);
            } else {
                if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                    FATAL("Unable to set memory permissions");
                block.state(ReadOnly);
            }
            break;
        case Dirty:
            FATAL("Block in incongruent state in acquire: %p", block.addr());
            break;
        case HostOnly:
            break;
    }
	return ret;
}
#endif

gmacError_t LazyBase::mapToAccelerator(Block &b)
{
    memory::StateBlock<State> &block = dynamic_cast<memory::StateBlock<State> &>(b);
    ASSERTION(block.state() == HostOnly);
    TRACE(LOCAL,"Mapping block to accelerator %p", block.addr());
    block.state(Dirty);
    addDirty(block);
    return gmacSuccess;
}

gmacError_t LazyBase::unmapFromAccelerator(Block &b)
{
    memory::StateBlock<State> &block = dynamic_cast<memory::StateBlock<State> &>(b);
    TRACE(LOCAL,"Unmapping block from accelerator %p", block.addr());
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

void LazyBase::addDirty(Block &block)
{
    dbl_.push(block);
    if(limit_ == unsigned(-1)) return;
    while(dbl_.size() > limit_) {
        Block *b = dbl_.pop();
        b->coherenceOp(&Protocol::release);
    }
    return;
}

gmacError_t LazyBase::releaseObjects()
{
    // We need to make sure that this operations is done before we
    // let other modes to proceed
    lock(); 
    while(dbl_.empty() == false) {
        Block *b = dbl_.pop();
        b->coherenceOp(&Protocol::release);
    }
    unlock();
    return gmacSuccess;
}

gmacError_t LazyBase::release(Block &b)
{
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    TRACE(LOCAL,"Releasing block %p", block.addr());
    gmacError_t ret = gmacSuccess;
    switch(block.state()) {
        case Dirty:
            ret = block.toAccelerator();
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

gmacError_t LazyBase::deleteBlock(Block &block)
{
    dbl_.remove(dynamic_cast<StateBlock<State> &>(block));
    return gmacSuccess;
}

gmacError_t LazyBase::toHost(Block &b)
{
    TRACE(LOCAL,"Sending block to host: %p", b.addr());
    gmacError_t ret = gmacSuccess;
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Invalid:
            TRACE(LOCAL,"Invalid block");
			if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            ret = block.toHost();
            if(ret != gmacSuccess) break;
            block.state(ReadOnly);
            break;
        case Dirty:
            TRACE(LOCAL,"Dirty block");
            break;
        case ReadOnly:
            TRACE(LOCAL,"ReadOnly block");
            break;
        case HostOnly:
            TRACE(LOCAL,"HostOnly block");
            break;
    }
    return ret;
}

gmacError_t LazyBase::toAccelerator(Block &b)
{
    TRACE(LOCAL,"Sending block to accelerator: %p", b.addr());
    gmacError_t ret = gmacSuccess;
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Dirty:
            TRACE(LOCAL,"Dirty block");
            ret = block.toAccelerator();
            if(ret != gmacSuccess) break;
            if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            block.state(ReadOnly);
            break;
        case Invalid:
            TRACE(LOCAL,"Invalid block");
            break;
        case ReadOnly:
            TRACE(LOCAL,"ReadOnly block");
            break;
        case HostOnly:
            TRACE(LOCAL,"HostOnly block");
            break;
    }
    return ret;
}

gmacError_t LazyBase::copyToBuffer(const Block &b, core::IOBuffer &buffer, size_t size,
							   unsigned bufferOffset, unsigned blockOffset) const
{
	gmacError_t ret = gmacSuccess;
	const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
	switch(block.state()) {
		case Invalid:
			ret = block.copyFromAccelerator(buffer, size, bufferOffset, blockOffset);
			break;
		case ReadOnly:
		case Dirty:
        case HostOnly:
			ret = block.copyFromHost(buffer, size, bufferOffset, blockOffset);
	}
	return ret;
}

gmacError_t LazyBase::copyFromBuffer(const Block &b, core::IOBuffer &buffer, size_t size, 
							   unsigned bufferOffset, unsigned blockOffset) const
{
	gmacError_t ret = gmacSuccess;
	const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
	switch(block.state()) {
		case Invalid:
			ret = block.copyToAccelerator(buffer, size, bufferOffset, blockOffset);
			break;
		case ReadOnly:
			ret = block.copyToAccelerator(buffer, size, bufferOffset, blockOffset);
			if(ret != gmacSuccess) break;
			ret = block.copyToHost(buffer, size, bufferOffset, blockOffset);
			break;
		case Dirty:			
        case HostOnly:
			ret = block.copyToHost(buffer, size, bufferOffset, blockOffset);
			break;
	}
	return ret;
}

gmacError_t LazyBase::memset(const Block &b, int v, size_t size, unsigned blockOffset) const
{
    gmacError_t ret = gmacSuccess;
	const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
	switch(block.state()) {
		case Invalid:
            ret = b.acceleratorMemset(v, size, blockOffset);
			break;
		case ReadOnly:
			ret = b.acceleratorMemset(v, size, blockOffset);
			if(ret != gmacSuccess) break;
			ret = b.hostMemset(v, size, blockOffset);
			break;
		case Dirty:			
        case HostOnly:
			ret = b.hostMemset(v, size, blockOffset);
			break;
	}
	return ret;
}

}}}
