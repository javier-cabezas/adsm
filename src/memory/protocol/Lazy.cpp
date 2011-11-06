#include "Lazy.h"

#include "core/io_buffer.h"

#include "config/config.h"

#include "memory/Memory.h"

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


LazyBase::LazyBase(bool eager) :
    gmac::util::mutex("LazyBase"),
    eager_(eager),
    limit_(1)
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


void LazyBase::deleteObject(object &obj)
{
    obj.decRef();
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

    if (block.getState() == lazy::Invalid) {
        ret = block.syncToHost();
        if(ret != gmacSuccess) goto exit_func;
        block.setState(lazy::ReadOnly);
    }

    if(block.protect(GMAC_PROT_READ) < 0)
        FATAL("Unable to set memory permissions");

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
    block.setState(lazy::Dirty, addr);
    block.unprotect();
    addDirty(block);
    TRACE(LOCAL,"Setting block %p to dirty state", block.addr());
    //ret = addDirty(block);
exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t LazyBase::acquire(Block &b, GmacProtection &prot)
{
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
    case lazy::ReadOnly:
        if (prot == GMAC_PROT_READWRITE ||
            prot == GMAC_PROT_WRITE) {
            if(block.protect(GMAC_PROT_NONE) < 0)
                FATAL("Unable to set memory permissions");
#ifndef USE_VM
            block.setState(lazy::Invalid);
            //block.acquired();
#endif
        }

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

void
LazyBase::addDirty(lazy::Block &block)
{
    lock();
    dbl_.push(block);
    if (eager_ == false) {
        unlock();
        return;
    } else {
        if (block.getCacheWriteFaults() >= __impl::util::params::ParamRollThreshold) {
            block.resetCacheWriteFaults();
            TRACE(LOCAL, "Increasing dirty block cache limit -> %u", limit_ + 1);
            limit_++;
        }
    }
    while (dbl_.size() > limit_) {
        Block &b = dbl_.front();
        b.coherenceOp(&Protocol::release);
    }
    unlock();
    return;
}

gmacError_t LazyBase::releaseAll()
{
    // We need to make sure that this operations is done before we
    // let other modes to proceed
    lock();

    // Shrink cache size if we have not filled it
    if (eager_ == true && dbl_.size() < limit_ && limit_ > 1) {
        limit_ /= 2;
    }

    // If the list of objects to be released is empty, assume a complete flush
    TRACE(LOCAL, "Releasing all blocks");

    while(dbl_.empty() == false) {
        Block &b = dbl_.front();
        gmacError_t ret = b.coherenceOp(&Protocol::release);
        ASSERTION(ret == gmacSuccess);
    }

    unlock();
    return gmacSuccess;
}

gmacError_t LazyBase::flushDirty()
{
    return releaseAll();
}


gmacError_t LazyBase::releasedAll()
{
    lock();

    // Shrink cache size if we have not filled it
    if (eager_ == true && dbl_.size() < limit_ && limit_ > 1) {
        TRACE(LOCAL, "Shrinking dirty block cache limit %u -> %u", limit_, limit_ / 2);
        limit_ /= 2;
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
        if(block.protect(GMAC_PROT_READ) < 0)
            FATAL("Unable to set memory permissions");
        ret = block.syncToAccelerator();
        if(ret != gmacSuccess) break;
        block.setState(lazy::ReadOnly);
        block.released();
        dbl_.remove(block);
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

#if 0
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

gmacError_t LazyBase::copyToBuffer(Block &b, core::io_buffer &buffer, size_t size,
                                   size_t bufferOff, size_t blockOff)
{
    gmacError_t ret = gmacSuccess;
    const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
        ret = block.copyToBuffer(buffer, bufferOff, blockOff, size, lazy::Block::ACCELERATOR);
        break;
    case lazy::ReadOnly:
    case lazy::Dirty:
    case lazy::HostOnly:
        ret = block.copyToBuffer(buffer, bufferOff, blockOff, size, lazy::Block::HOST);
        break;
    }
    return ret;
}

gmacError_t LazyBase::copyFromBuffer(Block &b, core::io_buffer &buffer, size_t size,
                                     size_t bufferOff, size_t blockOff)
{
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::ACCELERATOR);
        break;
    case lazy::ReadOnly:
#ifdef USE_OPENCL
        // WARNING: copying to host first because the io_buffer address can change in copyToAccelerator
        // if we do not wait
        ::memcpy(block.get_shadow() + blockOff, buffer.addr() + bufferOff, size);
        ret = block.owner().copy(block.get_device_addr() + ptroff_t(blockOff),
                                 buffer, bufferOff, size);
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::ACCELERATOR);
        if(ret != gmacSuccess) break;
#else
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::ACCELERATOR);
        if(ret != gmacSuccess) break;
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::HOST);
        if(ret != gmacSuccess) break;
#endif
        /* block.setState(lazy::Invalid); */
        break;
    case lazy::Dirty:
    case lazy::HostOnly:
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::HOST);
        break;
    }
    return ret;
}
#endif

hal::event_t
LazyBase::memset(const Block &b, size_t blockOffset, int v, size_t count, gmacError_t &err)
{
    hal::event_t ret;
    err = gmacSuccess;

    const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
        ret = block.owner().memset_async(block.get_device_addr() + blockOffset, v, count, err);
        break;
    case lazy::ReadOnly:
        ret = block.owner().memset_async(block.get_device_addr() + blockOffset, v, count, err);
        if(err != gmacSuccess) break;
        ::memset(block.get_shadow() + blockOffset, v, count);
        break;
    case lazy::Dirty:
    case lazy::HostOnly:
        ::memset(block.get_shadow() + blockOffset, v, count);
        break;
    }
    return ret;
}

#if 0
bool
LazyBase::isInAccelerator(Block &b)
{
    const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
    return block.getState() != lazy::Dirty;
}
#endif

hal::event_t
LazyBase::copyBlockToBlock(Block &d, size_t dstOff, Block &s, size_t srcOff, size_t count, gmacError_t &err)
{
    lazy::Block &dst = dynamic_cast<lazy::Block &>(d);
    lazy::Block &src = dynamic_cast<lazy::Block &>(s);

    err = gmacSuccess;

    hal::event_t ret;

    if ((src.getState() == lazy::Invalid || src.getState() == lazy::ReadOnly) &&
         dst.getState() == lazy::Invalid) {
        TRACE(LOCAL, "I || R -> I");
        // Copy acc-acc
        if (dst.owner().has_direct_copy(src.owner())) {
            ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                         src.get_device_addr() + srcOff, count, err);
        } else {
            ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                         src.get_shadow()      + srcOff, count, err);
        }
    } else if (src.getState() == lazy::Dirty && dst.getState() == lazy::Dirty) {
        // memcpy
        TRACE(LOCAL, "D -> D");
        ::memcpy(dst.get_shadow() + dstOff, src.get_shadow() + srcOff, count);
    } else if (src.getState() == lazy::ReadOnly &&
               dst.getState() == lazy::ReadOnly) {
        TRACE(LOCAL, "R -> R");
        // Copy acc-to-acc
        // memcpy
        if (dst.owner().has_direct_copy(src.owner())) {
            ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                         src.get_device_addr() + srcOff, count, err);
        } else {
            ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                         src.get_shadow()      + srcOff, count, err);
        }
        if (err == gmacSuccess) {
            ::memcpy(dst.get_shadow() + dstOff, src.get_shadow() + srcOff, count);
        }
    } else if (src.getState() == lazy::Invalid &&
               dst.getState() == lazy::ReadOnly) {
        TRACE(LOCAL, "I -> R");
        // Copy acc-to-acc
        if (dst.owner().has_direct_copy(src.owner())) {
            ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                         src.get_device_addr() + srcOff, count, err);
        } else {
            ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                         src.get_shadow()      + srcOff, count, err);
        }
        // acc-to-host
        if (err == gmacSuccess) {
            ret = src.owner().copy_async(dst.get_shadow()      + dstOff,
                                         src.get_device_addr() + srcOff, count, err);
        }
    } else if (src.getState() == lazy::Invalid &&
               dst.getState() == lazy::Dirty) {
        TRACE(LOCAL, "I -> D");
        // acc-to-host
        
        ret = src.owner().copy_async(dst.get_shadow()      + dstOff,
                                     src.get_device_addr() + srcOff, count, err);
    } else if (src.getState() == lazy::Dirty &&
               dst.getState() == lazy::Invalid) {
        TRACE(LOCAL, "D -> I");
        // host-to-acc
        ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                     src.get_shadow()      + srcOff, count, err);
    } else if (src.getState() == lazy::Dirty &&
               dst.getState() == lazy::ReadOnly) {
        // host-to-acc
        ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                     src.get_shadow()      + srcOff, count, err);
        TRACE(LOCAL, "D -> R");
        // host-to-host
        if (err == gmacSuccess) {
            ::memcpy(dst.get_shadow() + dstOff, src.get_shadow() + srcOff, count);
        }
    } else if (src.getState() == lazy::ReadOnly && dst.getState() == lazy::Dirty) {
        TRACE(LOCAL, "R -> D");
        // host-to-host
        ::memcpy(dst.get_shadow() + dstOff, src.get_shadow() + srcOff, count);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

hal::event_t
LazyBase::copyToBlock(Block &d, size_t dstOff,
                      hostptr_t src,
                      size_t count, gmacError_t &err)
{
    lazy::Block &dst = dynamic_cast<lazy::Block &>(d);

    err = gmacSuccess;

    hal::event_t ret;

    if (dst.getState() == lazy::Invalid) {
        TRACE(LOCAL, "-> I");
        // Copy acc-acc
        ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                     src, count, err);
    } else if (dst.getState() == lazy::Dirty) {
        // memcpy
        TRACE(LOCAL, "-> D");
        ::memcpy(dst.get_shadow() + dstOff, src, count);
    } else if (dst.getState() == lazy::ReadOnly) {
        TRACE(LOCAL, "-> R");
        // Copy acc-to-acc
        ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                     src, count, err);
        // memcpy
        ::memcpy(dst.get_shadow() + dstOff, src, count);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

hal::event_t
LazyBase::copyFromBlock(hostptr_t dst,
                        Block &s, size_t srcOff,
                        size_t count, gmacError_t &err)
{
    lazy::Block &src = dynamic_cast<lazy::Block &>(s);

    err = gmacSuccess;

    hal::event_t ret;

    if (src.getState() == lazy::Invalid) {
        TRACE(LOCAL, "I ->");
        // Copy acc-acc
        ret = src.owner().copy_async(dst,
                                     src.get_device_addr() + srcOff, count, err);
    } else if (src.getState() == lazy::Dirty) {
        // memcpy
        TRACE(LOCAL, "D ->");
        ::memcpy(dst, src.get_shadow() + srcOff, count);
    } else if (src.getState() == lazy::ReadOnly) {
        TRACE(LOCAL, "R ->");
        // Copy acc-to-acc
        ret = src.owner().copy_async(dst,
                                     src.get_device_addr() + srcOff, count, err);
        // memcpy
        ::memcpy(dst, src.get_shadow() + srcOff, count);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

hal::event_t
LazyBase::to_io_device(hal::device_output &output,
                       Block &s, size_t srcOff,
                       size_t count, gmacError_t &err)
{
    lazy::Block &src = dynamic_cast<lazy::Block &>(s);

    err = gmacSuccess;

    hal::event_t ret;

    if (src.getState() == lazy::Invalid) {
        TRACE(LOCAL, "I ->");
        // Copy acc-device
        ret = src.owner().copy_async(output,
                                     src.get_device_addr() + srcOff, count, err);
    } else if (src.getState() == lazy::Dirty) {
        // write to device
        TRACE(LOCAL, "D ->");
        output.write(src.get_shadow() + srcOff, count);
    } else if (src.getState() == lazy::ReadOnly) {
        TRACE(LOCAL, "R ->");
        // Copy acc-to-acc
        ret = src.owner().copy_async(output,
                                     src.get_device_addr() + srcOff, count, err);
        // write to device
        output.write(src.get_shadow() + srcOff, count);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

hal::event_t
LazyBase::from_io_device(Block &d, size_t dstOff,
                         hal::device_input &input,
                         size_t count, gmacError_t &err)
{
    lazy::Block &dst = dynamic_cast<lazy::Block &>(d);

    err = gmacSuccess;

    hal::event_t ret;

    if (dst.getState() == lazy::Invalid) {
        TRACE(LOCAL, "-> I");
        // Copy acc-acc
        ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                     input, count, err);
    } else if (dst.getState() == lazy::Dirty) {
        // memcpy
        TRACE(LOCAL, "-> D");
        input.read(dst.get_shadow() + dstOff, count);
    } else if (dst.getState() == lazy::ReadOnly) {
        TRACE(LOCAL, "-> R");
        // Copy acc-to-acc
        ret = dst.owner().copy_async(dst.get_device_addr() + dstOff,
                                     input, count, err);
        // memcpy
        input.read(dst.get_shadow() + dstOff, count);
    }

    TRACE(LOCAL, "Finished");
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
