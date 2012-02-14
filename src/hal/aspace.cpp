#include "types.h"

namespace __impl { namespace hal { namespace detail {

const unsigned &aspace::MaxBuffersIn_  = config::params::HALInputBuffersPerContext;
const unsigned &aspace::MaxBuffersOut_ = config::params::HALOutputBuffersPerContext;

buffer *
aspace::get_input_buffer(size_t size, stream &stream, event_ptr event)
{
    buffer *buffer = mapFreeBuffersIn_.pop(size);

    if (buffer == NULL) {
        nBuffersIn_.lock();
        if (nBuffersIn_ < MaxBuffersIn_) {
            nBuffersIn_++;
            nBuffersIn_.unlock();

            gmacError_t err;
            buffer = alloc_buffer(size, GMAC_PROT_READ, stream, err);
            ASSERTION(err == gmacSuccess);
        } else {
            nBuffersIn_.unlock();
            buffer = mapUsedBuffersIn_.pop(size);
            buffer->wait();
        }
    } else {
        TRACE(LOCAL, "Reusing input buffer");
    }

    buffer->set_event(event);

    mapUsedBuffersIn_.push(buffer, buffer->get_size());

    return buffer;
}

buffer *
aspace::get_output_buffer(size_t size, stream &stream, event_ptr event)
{
    buffer *buffer = mapFreeBuffersOut_.pop(size);

    if (buffer == NULL) {
        if (nBuffersOut_ < MaxBuffersOut_) {
            gmacError_t err;

            buffer = alloc_buffer(size, GMAC_PROT_WRITE, stream, err);
            ASSERTION(err == gmacSuccess);
            nBuffersOut_++;
        } else {
            buffer = mapUsedBuffersOut_.pop(size);
            buffer->wait();
        }
    } else {
        TRACE(LOCAL, "Reusing output buffer");
    }

    buffer->set_event(event);

    mapUsedBuffersOut_.push(buffer, buffer->get_size());

    return buffer;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
