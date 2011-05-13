#include "api/opencl/IOBuffer.h"
#include "api/opencl/lite/Mode.h"
#include "api/opencl/Tracer.h"

#include <CL/cl.h>

namespace __impl { namespace opencl { namespace lite {

Mode::Mode(cl_context ctx, cl_uint numDevices, const cl_device_id *devices) :
    gmac::util::Lock("Mode"),
    context_(ctx),
    active_(0),
    map_("ObjectMap")
{
    AtomicInc(Count_);
    // Create one command queue per device in the context
    cl_int ret = CL_SUCCESS;
    for(unsigned i = 0; i < numDevices; i++) {
        cl_command_queue queue = clCreateCommandQueue(ctx, devices[i], 0, &ret);
        ASSERTION(ret == CL_SUCCESS);
        streams_.insert(StreamMap::value_type(queue, devices[i]));
    }
}

Mode::~Mode()
{
    // If process is not valid, OpenCL might have been unloaded
    StreamMap::const_iterator i;
    cl_int ret = CL_SUCCESS;
    for(i = streams_.begin(); i != streams_.end(); i++) {
        ret = clReleaseCommandQueue(i->first);
        ASSERTION(ret == CL_SUCCESS);
    }
}

void Mode::addQueue(cl_command_queue queue)
{
    queues_.insert(queue);
}

gmacError_t Mode::setActiveQueue(cl_command_queue queue)
{
    bool valid = queues_.exists(queue);
    if(valid == false) return gmacErrorInvalidValue;
    active_ = queue;
    lock();
    return gmacSuccess;
}

void Mode::deactivateQueue()
{
    /* active_ = cl_command_queue(0); */
    unlock();
}

void Mode::removeQueue(cl_command_queue queue)
{
    if(active_ == queue) active_ = cl_command_queue(0);
    queues_.remove(queue);
}

gmacError_t Mode::map(accptr_t &dst, hostptr_t src, size_t size, unsigned align)
{
    cl_int ret = CL_SUCCESS;
    dst.base_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, NULL, &ret);
    dst.offset_ = 0;
    allocations_.insert(src, dst, size);
    return error(ret);
}

gmacError_t Mode::unmap(hostptr_t host, size_t size)
{
    ASSERTION(host != NULL);
    size_t s;
    const accptr_t &addr = allocations_.find(host, s);
    ASSERTION(addr != nullaccptr);
    ASSERTION(s == size);
    cl_int ret = CL_SUCCESS;
    ret = clReleaseMemObject(addr.base_);
    allocations_.erase(host, size);
    return error(ret);
}


gmacError_t Mode::bufferToAccelerator(accptr_t dst, core::IOBuffer &buffer, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to device %p ("FMT_SIZE" bytes)", buffer.addr(), dst.base_, len);
    gmacError_t ret = gmacSuccess;
    return ret;
}

gmacError_t Mode::acceleratorToBuffer(core::IOBuffer &buffer, const accptr_t src, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", src.base_, buffer.addr(), len);
    gmacError_t ret = gmacSuccess;
    return ret;
}


gmacError_t Mode::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Copy to accelerator: %p ("FMT_SIZE") @ %p", host, size, acc.base_);
    trace::SetThreadState(trace::Wait);
    StreamMap::const_iterator i;
    cl_event event;
    cl_int ret = CL_SUCCESS;
    for(i = streams_.begin(); i != streams_.end(); i++) {
        ret = clEnqueueWriteBuffer(i->first, acc.base_,
            CL_TRUE, acc.offset_, size, host, 0, NULL, &event);
        CFATAL(ret == CL_SUCCESS, "Error copying to accelerator: %d", ret);
        trace::SetThreadState(trace::Running);
        DataCommToAccelerator(*this, event, size);
        ret = clReleaseEvent(event);
        if(ret != CL_SUCCESS) goto do_exit;
    }
    for(i = streams_.begin(); i != streams_.end(); i++) {
        ret = clFinish(i->first);
        if(ret != CL_SUCCESS) goto do_exit;
    }
do_exit:
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Mode::copyToHost(hostptr_t host, const accptr_t acc, size_t count)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Copy to host: %p ("FMT_SIZE") @ %p", host, count, acc.base_);
    trace::SetThreadState(trace::Wait);
    cl_event event;
    cl_int ret = clEnqueueReadBuffer(active_, acc.base_,
        CL_TRUE, acc.offset_, count, host, 0, NULL, &event);
    CFATAL(ret == CL_SUCCESS, "Error copying to host: %d", ret);
    trace::SetThreadState(trace::Running);
    DataCommToAccelerator(*this, event, count);
    ret = clReleaseEvent(event);
    ASSERTION(ret == CL_SUCCESS);
    ret = clFinish(active_);
    ASSERTION(ret == CL_SUCCESS);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Mode::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Copy accelerator-accelerator ("FMT_SIZE") @ %p - %p", size,
        src.base_, dst.base_);
    // TODO: This is a very inefficient implementation. We might consider
    // using a kernel for this task
    void *tmp = ::malloc(size);

    StreamMap::const_iterator i;
    cl_int ret = CL_SUCCESS;
    for(i = streams_.begin(); i != streams_.end(); i++) {
        ret = clEnqueueReadBuffer(i->first, src.base_, CL_TRUE,
            src.offset_, size, tmp, 0, NULL, NULL);
        CFATAL(ret == CL_SUCCESS, "Error copying to host: %d", ret);
        if(ret == CL_SUCCESS) {
            ret = clEnqueueWriteBuffer(i->first, dst.base_, CL_TRUE,
                    dst.offset_, size, tmp, 0, NULL, NULL);
            CFATAL(ret == CL_SUCCESS, "Error copying to device: %d", ret);
    }
    }
    ::free(tmp);
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Mode::memset(accptr_t addr, int c, size_t size)
{
    trace::EnterCurrentFunction();
    // TODO: This is a very inefficient implementation. We might consider
    // using a kernel for this task
    void *tmp = ::malloc(size);
    ::memset(tmp, c, size);
    StreamMap::const_iterator i;
    cl_int ret = CL_SUCCESS;
    for(i = streams_.begin(); i != streams_.end(); i++) {
        ret = clEnqueueWriteBuffer(i->first, addr.base_,
            CL_TRUE, addr.offset_, size, tmp, 0, NULL, NULL);
    }
    ::free(tmp);
    trace::ExitCurrentFunction();
    return error(ret);
}

void Mode::memInfo(size_t &free, size_t &total)
{
    cl_int ret = CL_SUCCESS;
    cl_ulong value = 0;
    StreamMap::const_iterator i;
    free = total = size_t(-1);
    for(i = streams_.begin(); i != streams_.end(); i++) {
        ret = clGetDeviceInfo(i->second, CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(value), &value, NULL);
        CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);
        total = (total < size_t(value)) ? total : size_t(value);

        // TODO: This is actually wrong, but OpenCL do not let us know the
        // amount of free memory in the accelerator
        ret = clGetDeviceInfo(i->second, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
            sizeof(value), &value, NULL);
        CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);
        free = (free < size_t(value)) ? free : size_t(value);
    }
}


}}}
