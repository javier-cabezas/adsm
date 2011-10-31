#include "trace/Tracer.h"

#include "core/io_buffer.h"
#include "core/hpe/context.h"
#include "util/Logger.h"

namespace __impl { namespace core { namespace hpe {

context::context(hal::context_t &ctx,
                 hal::stream_t &streamLaunch,
                 hal::stream_t &streamToAccelerator,
                 hal::stream_t &streamToHost,
                 hal::stream_t &streamAccelerator) :
    gmac::util::RWLock("context"),
    ctx_(ctx),
    streamLaunch_(streamLaunch),
    streamToAccelerator_(streamToAccelerator),
    streamToHost_(streamToHost),
    streamAccelerator_(streamAccelerator),
    bufferWrite_(NULL),
    bufferRead_(NULL)
{
}

context::~context()
{
}

void
context::init()
{
}

gmacError_t context::copy(accptr_t acc, const hostptr_t host, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from host %p to accelerator %p", size, host, acc.get());
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(bufferWrite_ == NULL) bufferWrite_ = new gmac::core::io_buffer(ctx_, util::params::ParamBlockSize, GMAC_PROT_WRITE);
    if(bufferWrite_->async() == false) {
        delete bufferWrite_;
        bufferWrite_ = NULL;
        TRACE(LOCAL,"Not using pinned memory for transfer");
        hal::event_t &event = ctx_.copy(acc, host, size, streamToAccelerator_);
        gmacError_t ret = event.get_error();
        trace::ExitCurrentFunction();
        return ret;
    }
    gmacError_t ret = bufferWrite_->wait();
    ptroff_t offset = 0;
    while(size_t(offset) < size) {
        ret = bufferWrite_->wait();
        if(ret != gmacSuccess) break;
        ptroff_t len = ptroff_t(bufferWrite_->size());
        if((size - offset) < bufferWrite_->size()) len = ptroff_t(size - offset);
        trace::EnterCurrentFunction();
        ::memcpy(bufferWrite_->addr(), host + offset, len);
        trace::ExitCurrentFunction();
        ASSERTION(size_t(len) <= util::params::ParamBlockSize);
        hal::async_event_t &event = ctx_.copy_async(acc + offset, bufferWrite_->get_buffer(), 0, len, streamToAccelerator_);
        bufferWrite_->to_device(event);
        ret = event.get_error();
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        offset += len;
    }

    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t context::copy(hostptr_t host, const accptr_t acc, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from accelerator %p to host %p", size, acc.get(), host);
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess;
    if(bufferRead_ == NULL) bufferRead_ = new gmac::core::io_buffer(ctx_, util::params::ParamBlockSize, GMAC_PROT_READ);
    if(bufferRead_->async() == false) {
        delete bufferRead_;
        bufferRead_ = NULL;
        TRACE(LOCAL,"Not using pinned memory for transfer");
        hal::event_t &event = ctx_.copy(host, acc, size, streamToHost_);
        gmacError_t ret = event.get_error();
        trace::ExitCurrentFunction();
        return ret;
    }

    gmacError_t ret = bufferRead_->wait();
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    ptroff_t offset = 0;
    while(size_t(offset) < size) {
        ptroff_t len = ptroff_t(bufferRead_->size());
        if((size - offset) < bufferRead_->size()) len = ptroff_t(size - offset);
        hal::async_event_t &event = ctx_.copy_async(bufferRead_->get_buffer(), 0, acc + offset, len, streamToHost_);
        bufferRead_->to_host(event);
        ret = event.get_error();
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        ret = bufferRead_->wait();
        if(ret != gmacSuccess) break;
        trace::EnterCurrentFunction();
        ::memcpy((uint8_t *)host + offset, bufferRead_->addr(), len);
        trace::ExitCurrentFunction();
        offset += len;
    }
    trace::ExitCurrentFunction();
    return ret;
}


gmacError_t
context::copy(accptr_t dst, const accptr_t src, size_t count)
{
    TRACE(LOCAL,"Copy device %p to device %p ("FMT_SIZE" bytes)", src.get(), dst.get(), count);
    trace::EnterCurrentFunction();
    //gmacError_t ret = getAccelerator().copyToAcceleratorAsync(dst, buffer, off, len, *this, *streamToAccelerator_);
    hal::event_t &event = ctx_.copy(dst,
                                    src,
                                    count,
                                    streamAccelerator_);
    trace::ExitCurrentFunction();
    return event.get_error();

}

gmacError_t
context::copy(accptr_t dst, core::io_buffer &buffer, size_t off, size_t count)
{
    TRACE(LOCAL,"Copy %p to device %p ("FMT_SIZE" bytes)", buffer.addr(), dst.get(), count);
    trace::EnterCurrentFunction();
    //gmacError_t ret = getAccelerator().copyToAcceleratorAsync(dst, buffer, off, len, *this, *streamToAccelerator_);
    hal::async_event_t &event = ctx_.copy_async(dst,
                                                buffer.get_buffer(), off,
                                                count,
                                                streamToAccelerator_);
    buffer.to_device(event);
    trace::ExitCurrentFunction();
    return event.get_error();
}

gmacError_t
context::copy(core::io_buffer &buffer, size_t off, const accptr_t src, size_t count)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", src.get(), buffer.addr(), count);
    trace::EnterCurrentFunction();
    // Implement a function to remove these casts
    //gmacError_t ret = getAccelerator().copyToHostAsync(buffer, off, src, len, *this, *streamToHost_);
    hal::async_event_t &event = ctx_.copy_async(buffer.get_buffer(), off,
                                                src,
                                                count,
                                                streamToHost_);
    buffer.to_host(event);
    trace::ExitCurrentFunction();
    return event.get_error();
}

gmacError_t
context::memset(accptr_t addr, int c, size_t count)
{
    hal::event_t &event = ctx_.memset(addr,
                                      c,
                                      count,
                                      streamToAccelerator_);

    return event.get_error();
}

}}}
