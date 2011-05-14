#include "Lazy.h"

#include "core/IOBuffer.h"

#include "config/config.h"

#include "memory/Memory.h"
#include "memory/StateBlock.h"

#include "trace/Tracer.h"

#ifdef DEBUG
#include <ostream>
#endif


#if defined(__GNUC__)
#define MIN std::min
#elif defined(_MSC_VER)
#define MIN min
#endif

namespace __impl { namespace memory { namespace protocol {


LazyBase::LazyBase(size_t limit) :
    gmac::util::Lock("Lazy"),
    limit_(limit)
{
}

LazyBase::~LazyBase()
{

}

lazy::State LazyBase::state(GmacProtection prot) const
{
	switch(prot) {
		case GMAC_PROT_NONE: 
			return lazy::Invalid;
		case GMAC_PROT_READ:
			return lazy::ReadOnly;
		case GMAC_PROT_WRITE:
		case GMAC_PROT_READWRITE:
			return lazy::Dirty;
	}
	return lazy::Dirty;
}


void LazyBase::deleteObject(Object &obj)
{
    obj.release();
}



bool LazyBase::needUpdate(const Block &b) const
{
    const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
    switch(block.getState()) {        
        case lazy::Dirty:
        case lazy::HostOnly:
            return false;
        case lazy::ReadOnly:
        case lazy::Invalid:
            return true;
    }
    return false;
}

gmacError_t LazyBase::signalRead(Block &b, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    gmacError_t ret = gmacSuccess;

    block.read(addr);
    if(block.getState() == lazy::HostOnly) {
        WARNING("Signal on HostOnly block - Changing protection and continuing");
        if(block.unprotect() < 0)
            FATAL("Unable to set memory permissions");
        
        goto exit_func;
    }

    if (block.getState() != lazy::Invalid) {
        goto exit_func; // Somebody already fixed it
    }

    ret = block.syncToHost();
    if(ret != gmacSuccess) goto exit_func;
    if(block.protect(GMAC_PROT_READ) < 0)
        FATAL("Unable to set memory permissions");
    block.setState(lazy::ReadOnly);

exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t LazyBase::signalWrite(Block &b, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    gmacError_t ret = gmacSuccess;

    block.write(addr);
    switch (block.getState()) {
        case lazy::Dirty:            
            block.unprotect();
            goto exit_func; // Somebody already fixed it
        case lazy::Invalid:          
            ret = block.syncToHost();
            if(ret != gmacSuccess) goto exit_func;
            break;
        case lazy::HostOnly:
            WARNING("Signal on HostOnly block - Changing protection and continuing");
        case lazy::ReadOnly:
            break;
    }
    block.unprotect();
    block.setState(lazy::Dirty);
    addDirty(block);
    TRACE(LOCAL,"Setting block %p to dirty state", block.addr());
    //ret = addDirty(block);
exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t LazyBase::acquire(Block &b)
{
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
        case lazy::Invalid:
        case lazy::ReadOnly:
            if(block.protect(GMAC_PROT_NONE) < 0)
                FATAL("Unable to set memory permissions");
#ifndef USE_VM
            block.setState(lazy::Invalid);
#endif
            break;
        case lazy::Dirty:
            WARNING("Block modified before gmacSynchronize: %p", block.addr());
            break;
        case lazy::HostOnly:
            break;
    }
	return ret;
}

#ifdef USE_VM
gmacError_t LazyBase::acquireWithBitmap(Block &b)
{
    /// \todo Change this to the new BlockState 
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
        case lazy::Invalid:
        case lazy::ReadOnly:
            if (block.is(lazy::Invalid)) {
                if(block.protect(GMAC_PROT_NONE) < 0)
                    FATAL("Unable to set memory permissions");
                block.setState(lazy::Invalid);
            } else {
                if(block.protect(GMAC_PROT_READ) < 0)
                    FATAL("Unable to set memory permissions");
                block.setState(lazy::ReadOnly);
            }
            break;
        case lazy::Dirty:
            FATAL("Block in incongruent state in acquire: %p", block.addr());
            break;
        case lazy::HostOnly:
            break;
    }
	return ret;
}
#endif

gmacError_t LazyBase::mapToAccelerator(Block &b)
{
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    ASSERTION(block.getState() == lazy::HostOnly);
    TRACE(LOCAL,"Mapping block to accelerator %p", block.addr());
    block.setState(lazy::Dirty);
    addDirty(block);
    return gmacSuccess;
}

gmacError_t LazyBase::unmapFromAccelerator(Block &b)
{
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    TRACE(LOCAL,"Unmapping block from accelerator %p", block.addr());
    gmacError_t ret = gmacSuccess;
    switch(block.getState()) {
        case lazy::HostOnly:
        case lazy::Dirty:
        case lazy::ReadOnly:
            break;
        case lazy::Invalid:
            ret = block.syncToHost();
            if(ret != gmacSuccess) break;
    }
    if(block.unprotect() < 0)
        FATAL("Unable to set memory permissions");
    block.setState(lazy::HostOnly);
    dbl_.remove(block);
    return ret;
}

void LazyBase::addDirty(Block &block)
{
    lock(); 
    dbl_.push(block);
    if(limit_ == size_t(-1)) {
        unlock();
        return;
    }
    while(dbl_.size() > limit_) {
        Block &b = dbl_.pop();
        b.coherenceOp(&Protocol::release);
    }
    unlock();
    return;
}

gmacError_t LazyBase::releaseObjects()
{
    // We need to make sure that this operations is done before we
    // let other modes to proceed
    lock(); 
    while(dbl_.empty() == false) {
        Block &b = dbl_.pop();
        b.coherenceOp(&Protocol::release);
    }
    unlock();
    return gmacSuccess;
}

gmacError_t LazyBase::release(Block &b)
{
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    TRACE(LOCAL,"Releasing block %p", block.addr());
    gmacError_t ret = gmacSuccess;
    switch(block.getState()) {
        case lazy::Dirty:
            ret = block.syncToAccelerator();
            if(ret != gmacSuccess) break;
			if(block.protect(GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            block.setState(lazy::ReadOnly);
            break;
        case lazy::Invalid:
        case lazy::ReadOnly:
        case lazy::HostOnly:
            break;
    }
    return ret;
}

gmacError_t LazyBase::deleteBlock(Block &block)
{
    dbl_.remove(dynamic_cast<lazy::Block &>(block));
    return gmacSuccess;
}

gmacError_t LazyBase::toHost(Block &b)
{
    TRACE(LOCAL,"Sending block to host: %p", b.addr());
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
        case lazy::Invalid:
            ret = block.syncToHost();
            TRACE(LOCAL,"Invalid block");
			if(block.protect(GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            if(ret != gmacSuccess) break;
            block.setState(lazy::ReadOnly);
            break;
        case lazy::Dirty:
            TRACE(LOCAL,"Dirty block");
            break;
        case lazy::ReadOnly:
            TRACE(LOCAL,"ReadOnly block");
            break;
        case lazy::HostOnly:
            TRACE(LOCAL,"HostOnly block");
            break;
    }
    return ret;
}

gmacError_t LazyBase::toAccelerator(Block &b)
{
    TRACE(LOCAL,"Sending block to accelerator: %p", b.addr());
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
        case lazy::Dirty:
            TRACE(LOCAL,"Dirty block");
            if(block.protect(GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            ret = block.syncToAccelerator();
            if(ret != gmacSuccess) break;
            block.setState(lazy::ReadOnly);
            break;
        case lazy::Invalid:
            TRACE(LOCAL,"Invalid block");
            break;
        case lazy::ReadOnly:
            TRACE(LOCAL,"ReadOnly block");
            break;
        case lazy::HostOnly:
            TRACE(LOCAL,"HostOnly block");
            break;
    }
    return ret;
}

gmacError_t LazyBase::copyToBuffer(Block &b, core::IOBuffer &buffer, size_t size,
							   size_t bufferOffset, size_t blockOffset)
{
	gmacError_t ret = gmacSuccess;
	const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
	switch(block.getState()) {
		case lazy::Invalid:
			ret = block.copyFromAccelerator(buffer, size, bufferOffset, blockOffset);
			break;
		case lazy::ReadOnly:
		case lazy::Dirty:
        case lazy::HostOnly:
			ret = block.copyFromHost(buffer, size, bufferOffset, blockOffset);
	}
	return ret;
}

gmacError_t LazyBase::copyFromBuffer(Block &b, core::IOBuffer &buffer, size_t size, 
							   size_t bufferOffset, size_t blockOffset)
{
	gmacError_t ret = gmacSuccess;
	lazy::Block &block = dynamic_cast<lazy::Block &>(b);
	switch(block.getState()) {
		case lazy::Invalid:
			ret = block.copyToAccelerator(buffer, size, bufferOffset, blockOffset);
			break;
		case lazy::ReadOnly:
			ret = block.copyToAccelerator(buffer, size, bufferOffset, blockOffset);
			if(ret != gmacSuccess) break;
			/* ret = block.copyToHost(buffer, size, bufferOffset, blockOffset); */
            block.setState(lazy::Invalid);
			break;
		case lazy::Dirty:			
        case lazy::HostOnly:
			ret = block.copyToHost(buffer, size, bufferOffset, blockOffset);
			break;
	}
	return ret;
}

gmacError_t LazyBase::memset(const Block &b, int v, size_t size, size_t blockOffset)
{
    gmacError_t ret = gmacSuccess;
	const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
	switch(block.getState()) {
		case lazy::Invalid:
            ret = b.acceleratorMemset(v, size, blockOffset);
			break;
		case lazy::ReadOnly:
			ret = b.acceleratorMemset(v, size, blockOffset);
			if(ret != gmacSuccess) break;
			ret = b.hostMemset(v, size, blockOffset);
			break;
		case lazy::Dirty:			
        case lazy::HostOnly:
			ret = b.hostMemset(v, size, blockOffset);
			break;
	}
	return ret;
}

gmacError_t LazyBase::dump(Block &b, std::ostream &out, common::Statistic stat)
{
	lazy::BlockState &block = dynamic_cast<lazy::BlockState &>(b);
    //std::ostream *stream = (std::ostream *)param;
    //ASSERTION(stream != NULL);
    block.dump(out, stat);
    return gmacSuccess;
}

}}}
