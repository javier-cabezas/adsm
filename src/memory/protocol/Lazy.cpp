#include "Lazy.h"

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


lazy_base::lazy_base(bool eager) :
    Lock("LazyBase"),
    eager_(eager),
    limit_(1)
{
}

lazy_base::~lazy_base()
{
}

lazy_types::State lazy_base::state(GmacProtection prot) const
{
    switch(prot) {
    case GMAC_PROT_NONE:
        return lazy_types::Invalid;
    case GMAC_PROT_READ:
        return lazy_types::ReadOnly;
    case GMAC_PROT_WRITE:
    case GMAC_PROT_READWRITE:
            return lazy_types::Dirty;
    }
    return lazy_types::Dirty;
}


void lazy_base::deleteObject(object &obj)
{
    obj.decRef();
}

bool lazy_base::needUpdate(const block_ptr b) const
{
    const lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    switch(block->getState()) {
    case lazy_types::Dirty:
    case lazy_types::HostOnly:
        return false;
    case lazy_types::ReadOnly:
    case lazy_types::Invalid:
        return true;
    }
    return false;
}

gmacError_t lazy_base::signal_read(block_ptr b, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    gmacError_t ret = gmacSuccess;

    block->read(addr);
    if(block->getState() == lazy_types::HostOnly) {
        WARNING("Signal on HostOnly block - Changing protection and continuing");
        if (block->unprotect() < 0)
            FATAL("Unable to set memory permissions");

        goto exit_func;
    }

    if (block->getState() == lazy_types::Invalid) {
        ret = block->syncToHost();
        if(ret != gmacSuccess) goto exit_func;
        block->setState(lazy_types::ReadOnly);
    }

    if(block->protect(GMAC_PROT_READ) < 0)
        FATAL("Unable to set memory permissions");

exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t lazy_base::signal_write(block_ptr b, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    gmacError_t ret = gmacSuccess;

    block->write(addr);
    switch (block->getState()) {
    case lazy_types::Dirty:
        block->unprotect();
        goto exit_func; // Somebody already fixed it
    case lazy_types::Invalid:
        ret = block->syncToHost();
        if(ret != gmacSuccess) goto exit_func;
        break;
    case lazy_types::HostOnly:
        WARNING("Signal on HostOnly block - Changing protection and continuing");
    case lazy_types::ReadOnly:
        break;
    }
    block->setState(lazy_types::Dirty, addr);
    block->unprotect();
    addDirty(block);
    TRACE(LOCAL,"Setting block %p to dirty state", block->addr());
    //ret = addDirty(block);
exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t lazy_base::acquire(block_ptr b, GmacProtection &prot)
{
    gmacError_t ret = gmacSuccess;
    lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    switch(block->getState()) {
    case lazy_types::Invalid:
    case lazy_types::ReadOnly:
        if (prot == GMAC_PROT_READWRITE ||
            prot == GMAC_PROT_WRITE) {
            if(block->protect(GMAC_PROT_NONE) < 0)
                FATAL("Unable to set memory permissions");
#ifndef USE_VM
            block->setState(lazy_types::Invalid);
            //block->acquired();
#endif
        }

        break;
    case lazy_types::Dirty:
        WARNING("Block modified before gmacSynchronize: %p", block->addr());
        break;
    case lazy_types::HostOnly:
        break;
    }
    return ret;
}

#ifdef USE_VM
gmacError_t lazy_base::acquireWithBitmap(block_ptr b)
{
    /// \todo Change this to the new BlockState
    gmacError_t ret = gmacSuccess;
    lazy_types::block_ptr block = util::smart_ptr<lazy_types::block>::static_pointer_cast(b);
    switch(block->getState()) {
    case lazy_types::Invalid:
    case lazy_types::ReadOnly:
        if (block->is(lazy_types::Invalid)) {
            if(block->protect(GMAC_PROT_NONE) < 0)
                FATAL("Unable to set memory permissions");
            block->setState(lazy_types::Invalid);
        } else {
            if(block->protect(GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            block->setState(lazy_types::ReadOnly);
        }
        break;
    case lazy_types::Dirty:
        FATAL("Block in incongruent state in acquire: %p", block->addr());
        break;
    case lazy_types::HostOnly:
        break;
    }
    return ret;
}
#endif

gmacError_t lazy_base::mapToAccelerator(block_ptr b)
{
    lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    ASSERTION(block->getState() == lazy_types::HostOnly);
    TRACE(LOCAL,"Mapping block to accelerator %p", block->addr());
    block->setState(lazy_types::Dirty);
    addDirty(block);
    return gmacSuccess;
}

gmacError_t lazy_base::unmapFromAccelerator(block_ptr b)
{
    lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    TRACE(LOCAL,"Unmapping block from accelerator %p", block->addr());
    gmacError_t ret = gmacSuccess;
    switch(block->getState()) {
    case lazy_types::HostOnly:
    case lazy_types::Dirty:
    case lazy_types::ReadOnly:
        break;
    case lazy_types::Invalid:
        ret = block->syncToHost();
        if(ret != gmacSuccess) break;
    }
    if(block->unprotect() < 0)
        FATAL("Unable to set memory permissions");
    block->setState(lazy_types::HostOnly);
    dbl_.remove(block);
    return ret;
}

void
lazy_base::addDirty(lazy_types::block_ptr block)
{
    lock();
    dbl_.push(block);
    if (eager_ == false) {
        unlock();
        return;
    } else {
        if (block->getCacheWriteFaults() >= config::params::RollThreshold) {
            block->resetCacheWriteFaults();
            TRACE(LOCAL, "Increasing dirty block cache limit -> %u", limit_ + 1);
            limit_++;
        }
    }
    while (dbl_.size() > limit_) {
        lazy_types::block_ptr b = util::static_pointer_cast<lazy_types::Block>(dbl_.front());
        release(b);
    }
    unlock();
    return;
}

gmacError_t lazy_base::releaseAll()
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
        block_ptr b = dbl_.front();
        gmacError_t ret = release(b);
        ASSERTION(ret == gmacSuccess);
    }

    unlock();
    return gmacSuccess;
}

gmacError_t lazy_base::flushDirty()
{
    return releaseAll();
}


gmacError_t lazy_base::releasedAll()
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

gmacError_t lazy_base::release(block_ptr b)
{
    lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    TRACE(LOCAL,"Releasing block %p", block->addr());
    gmacError_t ret = gmacSuccess;
    switch(block->getState()) {
    case lazy_types::Dirty:
        if(block->protect(GMAC_PROT_READ) < 0)
            FATAL("Unable to set memory permissions");
        ret = block->syncToAccelerator();
        if(ret != gmacSuccess) break;
        block->setState(lazy_types::ReadOnly);
        block->released();
        dbl_.remove(block);
        break;
    case lazy_types::Invalid:
    case lazy_types::ReadOnly:
    case lazy_types::HostOnly:
        break;
    }
    return ret;
}

gmacError_t lazy_base::deleteBlock(block_ptr block)
{
    dbl_.remove(block);
    return gmacSuccess;
}

gmacError_t lazy_base::toHost(block_ptr b)
{
    TRACE(LOCAL,"Sending block to host: %p", b->addr());
    gmacError_t ret = gmacSuccess;
    lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    switch(block->getState()) {
    case lazy_types::Invalid:
        ret = block->syncToHost();
        TRACE(LOCAL,"Invalid block");
        if(block->protect(GMAC_PROT_READ) < 0)
            FATAL("Unable to set memory permissions");
        if(ret != gmacSuccess) break;
        block->setState(lazy_types::ReadOnly);
        break;
    case lazy_types::Dirty:
        TRACE(LOCAL,"Dirty block");
        break;
    case lazy_types::ReadOnly:
        TRACE(LOCAL,"ReadOnly block");
        break;
    case lazy_types::HostOnly:
        TRACE(LOCAL,"HostOnly block");
        break;
    }
    return ret;
}

#if 0
gmacError_t lazy_base::toAccelerator(block_ptr b)
{
    TRACE(LOCAL,"Sending block to accelerator: %p", b->addr());
    gmacError_t ret = gmacSuccess;
    lazy_types::block_ptr block = util::smart_ptr<lazy_types::block>::static_pointer_cast(b);
    switch(block->getState()) {
    case lazy_types::Dirty:
        TRACE(LOCAL,"Dirty block");
        if(block->protect(GMAC_PROT_READ) < 0)
            FATAL("Unable to set memory permissions");
        ret = block->syncToAccelerator();
        if(ret != gmacSuccess) break;
        block->setState(lazy_types::ReadOnly);
        break;
    case lazy_types::Invalid:
        TRACE(LOCAL,"Invalid block");
        break;
    case lazy_types::ReadOnly:
        TRACE(LOCAL,"ReadOnly block");
        break;
    case lazy_types::HostOnly:
        TRACE(LOCAL,"HostOnly block");
        break;
    }
    return ret;
}

gmacError_t lazy_base::copyToBuffer(block_ptr b, core::io_buffer &buffer, size_t size,
                                   size_t bufferOff, size_t blockOff)
{
    gmacError_t ret = gmacSuccess;
    const lazy_types::block_ptr block = dynamic_cast<const lazy_types::block_ptr>(b);
    switch(block.getState()) {
    case lazy_types::Invalid:
        ret = block.copyToBuffer(buffer, bufferOff, blockOff, size, lazy_types::block::ACCELERATOR);
        break;
    case lazy_types::ReadOnly:
    case lazy_types::Dirty:
    case lazy_types::HostOnly:
        ret = block.copyToBuffer(buffer, bufferOff, blockOff, size, lazy_types::block::HOST);
        break;
    }
    return ret;
}

gmacError_t lazy_base::copyFromBuffer(block_ptr b, core::io_buffer &buffer, size_t size,
                                     size_t bufferOff, size_t blockOff)
{
    gmacError_t ret = gmacSuccess;
    lazy_types::block_ptr block = dynamic_cast<lazy_types::block_ptr>(b);
    switch(block.getState()) {
    case lazy_types::Invalid:
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy_types::block::ACCELERATOR);
        break;
    case lazy_types::ReadOnly:
#ifdef USE_OPENCL
        // WARNING: copying to host first because the io_buffer address can change in copyToAccelerator
        // if we do not wait
        ::memcpy(block.get_shadow() + blockOff, buffer.addr() + bufferOff, size);
        ret = block.owner().copy(block.get_device_addr() + ptroff_t(blockOff),
                                 buffer, bufferOff, size);
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy_types::block::ACCELERATOR);
        if(ret != gmacSuccess) break;
#else
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy_types::block::ACCELERATOR);
        if(ret != gmacSuccess) break;
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy_types::block::HOST);
        if(ret != gmacSuccess) break;
#endif
        /* block.setState(lazy::Invalid); */
        break;
    case lazy_types::Dirty:
    case lazy_types::HostOnly:
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy_types::block::HOST);
        break;
    }
    return ret;
}
#endif

hal::event_t
lazy_base::memset(const block_ptr b, size_t blockOffset, int v, size_t count, gmacError_t &err)
{
    hal::event_t ret;
    err = gmacSuccess;

    const lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    switch(block->getState()) {
    case lazy_types::Invalid:
        ret = block->owner()->memset_async(block->get_device_addr() + blockOffset, v, count, err);
        break;
    case lazy_types::ReadOnly:
        ret = block->owner()->memset_async(block->get_device_addr() + blockOffset, v, count, err);
        if(err != gmacSuccess) break;
        ::memset(block->get_shadow() + blockOffset, v, count);
        break;
    case lazy_types::Dirty:
    case lazy_types::HostOnly:
        ::memset(block->get_shadow() + blockOffset, v, count);
        break;
    }
    return ret;
}

#if 0
bool
lazy_base::isInAccelerator(block_ptr b)
{
    const lazy_types::block_ptr block = dynamic_cast<const lazy_types::block_ptr>(b);
    return block.getState() != lazy_types::Dirty;
}
#endif

hal::event_t
lazy_base::copyBlockToBlock(block_ptr d, size_t dstOff, block_ptr s, size_t srcOff, size_t count, gmacError_t &err)
{
    lazy_types::block_ptr dst = util::static_pointer_cast<lazy_types::Block>(d);
    lazy_types::block_ptr src = util::static_pointer_cast<lazy_types::Block>(s);

    err = gmacSuccess;

    hal::event_t ret;

    if ((src->getState() == lazy_types::Invalid || src->getState() == lazy_types::ReadOnly) &&
         dst->getState() == lazy_types::Invalid) {
        TRACE(LOCAL, "I || R -> I");
        // Copy acc-acc
        if (dst->owner()->has_direct_copy(src->owner())) {
            ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                           src->get_device_addr() + srcOff, count, err);
        } else {
            ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                           hal::ptr_t(src->get_shadow() + srcOff), count, err);
        }
    } else if (src->getState() == lazy_types::Dirty && dst->getState() == lazy_types::Dirty) {
        // memcpy
        TRACE(LOCAL, "D -> D");
        ::memcpy(dst->get_shadow() + dstOff, src->get_shadow() + srcOff, count);
    } else if (src->getState() == lazy_types::ReadOnly &&
               dst->getState() == lazy_types::ReadOnly) {
        TRACE(LOCAL, "R -> R");
        // Copy acc-to-acc
        // memcpy
        if (dst->owner()->has_direct_copy(src->owner())) {
            ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                           src->get_device_addr() + srcOff, count, err);
        } else {
            ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                           hal::ptr_t(src->get_shadow() + srcOff), count, err);
        }
        if (err == gmacSuccess) {
            ::memcpy(dst->get_shadow() + dstOff, src->get_shadow() + srcOff, count);
        }
    } else if (src->getState() == lazy_types::Invalid &&
               dst->getState() == lazy_types::ReadOnly) {
        TRACE(LOCAL, "I -> R");
        // Copy acc-to-acc
        if (dst->owner()->has_direct_copy(src->owner())) {
            ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                           src->get_device_addr() + srcOff, count, err);
        } else {
            ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                           hal::ptr_t(src->get_shadow() + srcOff), count, err);
        }
        // acc-to-host
        if (err == gmacSuccess) {
            ret = src->owner()->copy_async(hal::ptr_t(dst->get_shadow() + dstOff),
                                           src->get_device_addr() + srcOff, count, err);
        }
    } else if (src->getState() == lazy_types::Invalid &&
               dst->getState() == lazy_types::Dirty) {
        TRACE(LOCAL, "I -> D");
        // acc-to-host
        
        ret = src->owner()->copy_async(hal::ptr_t(dst->get_shadow() + dstOff),
                                       src->get_device_addr() + srcOff, count, err);
    } else if (src->getState() == lazy_types::Dirty &&
               dst->getState() == lazy_types::Invalid) {
        TRACE(LOCAL, "D -> I");
        // host-to-acc
        ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                       hal::ptr_t(src->get_shadow() + srcOff), count, err);
    } else if (src->getState() == lazy_types::Dirty &&
               dst->getState() == lazy_types::ReadOnly) {
        // host-to-acc
        ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                       hal::ptr_t(src->get_shadow() + srcOff), count, err);
        TRACE(LOCAL, "D -> R");
        // host-to-host
        if (err == gmacSuccess) {
            ::memcpy(dst->get_shadow() + dstOff, src->get_shadow() + srcOff, count);
        }
    } else if (src->getState() == lazy_types::ReadOnly && dst->getState() == lazy_types::Dirty) {
        TRACE(LOCAL, "R -> D");
        // host-to-host
        ::memcpy(dst->get_shadow() + dstOff, src->get_shadow() + srcOff, count);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

hal::event_t
lazy_base::copyToBlock(block_ptr d, size_t dstOff,
                      hostptr_t src,
                      size_t count, gmacError_t &err)
{
    lazy_types::block_ptr dst = util::static_pointer_cast<lazy_types::Block>(d);

    err = gmacSuccess;

    hal::event_t ret;

    if (dst->getState() == lazy_types::Invalid) {
        TRACE(LOCAL, "-> I");
        // Copy acc-acc
        ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                       hal::ptr_t(src), count, err);
    } else if (dst->getState() == lazy_types::Dirty) {
        // memcpy
        TRACE(LOCAL, "-> D");
        ::memcpy(dst->get_shadow() + dstOff, src, count);
    } else if (dst->getState() == lazy_types::ReadOnly) {
        TRACE(LOCAL, "-> R");
        // Copy acc-to-acc
        ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                       hal::ptr_t(src), count, err);
        // memcpy
        ::memcpy(dst->get_shadow() + dstOff, src, count);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

hal::event_t
lazy_base::copyFromBlock(hostptr_t dst,
                         block_ptr s, size_t srcOff,
                         size_t count, gmacError_t &err)
{
    lazy_types::block_ptr src = util::static_pointer_cast<lazy_types::Block>(s);

    err = gmacSuccess;

    hal::event_t ret;

    if (src->getState() == lazy_types::Invalid) {
        TRACE(LOCAL, "I ->");
        // Copy acc-acc
        ret = src->owner()->copy_async(hal::ptr_t(dst),
                                       src->get_device_addr() + srcOff, count, err);
    } else if (src->getState() == lazy_types::Dirty) {
        // memcpy
        TRACE(LOCAL, "D ->");
        ::memcpy(dst, src->get_shadow() + srcOff, count);
    } else if (src->getState() == lazy_types::ReadOnly) {
        TRACE(LOCAL, "R ->");
        // Copy acc-to-acc
        ret = src->owner()->copy_async(hal::ptr_t(dst),
                                       src->get_device_addr() + srcOff, count, err);
        // memcpy
        ::memcpy(dst, src->get_shadow() + srcOff, count);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

hal::event_t
lazy_base::to_io_device(hal::device_output &output,
                       block_ptr s, size_t srcOff,
                       size_t count, gmacError_t &err)
{
    lazy_types::block_ptr src = util::static_pointer_cast<lazy_types::Block>(s);

    err = gmacSuccess;

    hal::event_t ret;

    if (src->getState() == lazy_types::Invalid) {
        TRACE(LOCAL, "I ->");
        // Copy acc-disk
        ret = src->owner()->copy_async(output,
                                       src->get_device_addr() + srcOff, count, err);
    } else if (src->getState() == lazy_types::Dirty) {
        // write to device
        TRACE(LOCAL, "D ->");
        output.write(src->get_shadow() + srcOff, count);
    } else if (src->getState() == lazy_types::ReadOnly) {
        TRACE(LOCAL, "R ->");
        // Copy acc-to-disk
        ret = src->owner()->copy_async(output,
                                       src->get_device_addr() + srcOff, count, err);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

hal::event_t
lazy_base::from_io_device(block_ptr d, size_t dstOff,
                         hal::device_input &input,
                         size_t count, gmacError_t &err)
{
    lazy_types::block_ptr dst = util::static_pointer_cast<lazy_types::Block>(d);

    err = gmacSuccess;

    hal::event_t ret;

    if (dst->getState() == lazy_types::Invalid) {
        TRACE(LOCAL, "-> I");
        // Copy disk-acc
        ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
                                       input, count, err);
    } else if (dst->getState() == lazy_types::Dirty) {
        // memcpy
        TRACE(LOCAL, "-> D");
        input.read(dst->get_shadow() + dstOff, count);
    } else if (dst->getState() == lazy_types::ReadOnly) {
        TRACE(LOCAL, "-> R");
        // disk-to-host
        input.read(dst->get_shadow() + dstOff, count);

        // Copy host-to-acc
        ret = dst->owner()->copy_async(dst->get_device_addr() + dstOff,
        		                       hal::ptr_t(dst->get_shadow() + dstOff), count, err);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

gmacError_t lazy_base::dump(block_ptr b, std::ostream &out, common::Statistic stat)
{
    lazy_types::block_ptr block = util::static_pointer_cast<lazy_types::Block>(b);
    //std::ostream *stream = (std::ostream *)param;
    //ASSERTION(stream != NULL);
    //block->dump(out, stat);
    return gmacSuccess;
}

}}}
