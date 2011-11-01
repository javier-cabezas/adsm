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
    gmac::util::lock_rw("context"),
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

hal::event_t *
context::copy(accptr_t acc, const hostptr_t host, size_t size, gmacError_t &err)
{
    hal::event_t *ret = ctx_.copy(acc, host, size, streamToAccelerator_, err);

    return ret;
}

hal::async_event_t *
context::copy_async(accptr_t acc, const hostptr_t host, size_t size, gmacError_t &err)
{
    hal::async_event_t *ret = NULL;
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from host %p to accelerator %p", size, host, acc.get());
    if(size == 0) {
        err = gmacErrorInvalidValue;
        return NULL; /* Fast path */
    }
    trace::EnterCurrentFunction();
    /* In case there is no page-locked memory available, use the slow path */
    if(bufferWrite_ == NULL) bufferWrite_ = new gmac::core::io_buffer(ctx_, util::params::ParamBlockSize, GMAC_PROT_WRITE);
    if(bufferWrite_->async() == false) {
        delete bufferWrite_;
        bufferWrite_ = NULL;
        TRACE(LOCAL,"Not using pinned memory for transfer");
        trace::ExitCurrentFunction();
        return NULL;
    }

    ptroff_t offset = 0;
    while(size_t(offset) < size) {
        err = bufferWrite_->wait();
        if (err != gmacSuccess) break;
        ptroff_t len = ptroff_t(bufferWrite_->size());
        if ((size - offset) < bufferWrite_->size()) len = ptroff_t(size - offset);
        trace::EnterCurrentFunction();
        ::memcpy(bufferWrite_->addr(), host + offset, len);
        trace::ExitCurrentFunction();
        ASSERTION(size_t(len) <= util::params::ParamBlockSize);
        ret = ctx_.copy_async(acc + offset, bufferWrite_->get_buffer(), 0, len, streamToAccelerator_, err);
        ASSERTION(err == gmacSuccess);
        if(err != gmacSuccess) break;
        bufferWrite_->to_device(*ret);
        offset += len;
    }

    trace::ExitCurrentFunction();
    return ret;
}

hal::event_t *
context::copy(hostptr_t host, const accptr_t acc, size_t size, gmacError_t &err)
{
    hal::event_t *ret = ctx_.copy(host, acc, size, streamToHost_, err);

    return ret;
}

hal::async_event_t *
context::copy_async(hostptr_t host, const accptr_t acc, size_t size, gmacError_t &err)
{
    hal::async_event_t *ret = NULL;
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from accelerator %p to host %p", size, acc.get(), host);
    if(size == 0) {
        err = gmacErrorInvalidValue;
        return NULL; /* Fast path */
    }
    trace::EnterCurrentFunction();
    if(bufferRead_ == NULL) bufferRead_ = new gmac::core::io_buffer(ctx_, util::params::ParamBlockSize, GMAC_PROT_READ);
    if(bufferRead_->async() == false) {
        delete bufferRead_;
        bufferRead_ = NULL;
        TRACE(LOCAL,"Not using pinned memory for transfer");
        trace::ExitCurrentFunction();
        return NULL;
    }

    err = bufferRead_->wait();
    if(err != gmacSuccess) { trace::ExitCurrentFunction(); return NULL; }
    ptroff_t offset = 0;
    while(size_t(offset) < size) {
        ptroff_t len = ptroff_t(bufferRead_->size());
        if((size - offset) < bufferRead_->size()) len = ptroff_t(size - offset);
        ret = ctx_.copy_async(bufferRead_->get_buffer(), 0, acc + offset, len, streamToHost_, err);
        ASSERTION(err == gmacSuccess);
        if(err != gmacSuccess) break;
        bufferRead_->to_host(*ret);
        err = bufferRead_->wait();
        if(err != gmacSuccess) break;
        trace::EnterCurrentFunction();
        ::memcpy((uint8_t *)host + offset, bufferRead_->addr(), len);
        trace::ExitCurrentFunction();
        offset += len;
    }
    trace::ExitCurrentFunction();
    return ret;
}

hal::event_t *
context::copy(accptr_t dst, const accptr_t src, size_t count, gmacError_t &err)
{
    TRACE(LOCAL,"Copy device %p to device %p ("FMT_SIZE" bytes)", src.get(), dst.get(), count);
    trace::EnterCurrentFunction();

    hal::event_t *ret = ctx_.copy(dst,
                                  src,
                                  count,
                                  streamAccelerator_, err);
    trace::ExitCurrentFunction();
    return ret;

}

hal::async_event_t *
context::copy_async(accptr_t dst, const accptr_t src, size_t count, gmacError_t &err)
{
    TRACE(LOCAL,"Copy device %p to device %p ("FMT_SIZE" bytes)", src.get(), dst.get(), count);
    trace::EnterCurrentFunction();

    hal::async_event_t *ret = ctx_.copy_async(dst,
                                              src,
                                              count,
                                              streamAccelerator_, err);
    trace::ExitCurrentFunction();
    return ret;

}

gmacError_t
context::copy(accptr_t dst, core::io_buffer &buffer, size_t off, size_t count)
{
    TRACE(LOCAL,"Copy %p to device %p ("FMT_SIZE" bytes)", buffer.addr(), dst.get(), count);
    trace::EnterCurrentFunction();

    gmacError_t ret;

    hal::event_t *event = ctx_.copy(dst,
                                    buffer.addr() + off,
                                    count,
                                    streamToAccelerator_, ret);

    if (ret == gmacSuccess) {
        buffer.to_device(*event);
    }

    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
context::copy(core::io_buffer &buffer, size_t off, const accptr_t src, size_t count)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", src.get(), buffer.addr(), count);
    trace::EnterCurrentFunction();

    gmacError_t ret;

    hal::event_t *event = ctx_.copy(buffer.addr() + off,
                                    src,
                                    count,
                                    streamToHost_, ret);
    
    if (ret == gmacSuccess) {
        buffer.to_host(*event);
    }

    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
context::copy_async(accptr_t dst, core::io_buffer &buffer, size_t off, size_t count)
{
    if (buffer.async() == false) {
        return gmacErrorInvalidValue;
    }

    TRACE(LOCAL,"Copy %p to device %p ("FMT_SIZE" bytes)", buffer.addr(), dst.get(), count);
    trace::EnterCurrentFunction();

    gmacError_t ret;

    hal::async_event_t *event = ctx_.copy_async(dst,
                                                buffer.get_buffer(), off,
                                                count,
                                                streamToAccelerator_, ret);
    if (ret == gmacSuccess) {
        buffer.to_device(*event);
    }

    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
context::copy_async(core::io_buffer &buffer, size_t off, const accptr_t src, size_t count)
{
    if (buffer.async() == false) {
        return gmacErrorInvalidValue;
    }

    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", src.get(), buffer.addr(), count);
    trace::EnterCurrentFunction();

    gmacError_t ret;

    hal::async_event_t *event = ctx_.copy_async(buffer.get_buffer(), off,
                                                src,
                                                count,
                                                streamToHost_, ret);
    if (ret == gmacSuccess) {
        buffer.to_host(*event);
    }

    trace::ExitCurrentFunction();
    return ret;
}

hal::event_t *
context::memset(accptr_t addr, int c, size_t count, gmacError_t &err)
{
    hal::event_t *ret = ctx_.memset(addr,
                                    c,
                                    count,
                                    streamToAccelerator_, err);

    return ret;
}

hal::async_event_t *
context::memset_async(accptr_t addr, int c, size_t count, gmacError_t &err)
{
    hal::async_event_t *ret = ctx_.memset_async(addr,
                                                c,
                                                count,
                                                streamToAccelerator_, err);

    return ret;
}

}}}
