#include "Accelerator.h"
#include "Mode.h"

#include "core/Process.h"

namespace __impl { namespace opencl {

Accelerator::Accelerator(int n, cl_platform_id platform, cl_device_id device) :
    core::Accelerator(n), platform_(platform), device_(device)
{
    // Not used for now
    busId_ = 0;
    busAccId_ = 0;

    cl_bool val = CL_FALSE;
    cl_int ret = clGetDeviceInfo(device_, CL_DEVICE_HOST_UNIFIED_MEMORY,
        sizeof(val), &val, NULL);
    CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);
    integrated_ = (val == CL_TRUE);

    cl_context_properties prop[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_, NULL };

    ctx_ = clCreateContext(prop, 1, &device_, NULL, NULL, &ret);
    CFATAL(ret == CL_SUCCESS, "Unable to create OpenCL context %d", ret);

    cl_command_queue stream;
    stream = clCreateCommandQueue(ctx_, device_, 0, &ret);
    CFATAL(ret == CL_SUCCESS, "Unable to create OpenCL stream");
    cmd_.add(stream);
}

Accelerator::~Accelerator()
{
    ASSERTION(clReleaseContext(ctx_) == CL_SUCCESS);
}

void Accelerator::init()
{
}

core::Mode *Accelerator::createMode(core::Process &proc)
{
    trace::EnterCurrentFunction();
    core::Mode *mode = new opencl::Mode(proc, *this);
    if (mode != NULL) {
        registerMode(*mode);
    }
    trace::ExitCurrentFunction();

    TRACE(LOCAL,"Creating Execution Mode %p to Accelerator", mode);
    return mode;
}

gmacError_t Accelerator::malloc(accptr_t &addr, size_t size, unsigned align) 
{
    trace::EnterCurrentFunction();
    cl_int ret = CL_SUCCESS;
    addr.base_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, size, NULL, &ret);
    addr.offset_ = 0;
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::free(accptr_t addr)
{
    trace::EnterCurrentFunction();
    cl_int ret = clReleaseMemObject(addr.base_);
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy to accelerator: %p ("FMT_SIZE")", host, size);
    cl_int ret = clEnqueueWriteBuffer(cmd_.front(), acc.base_,
        CL_TRUE, acc.offset_, size, host, 0, NULL, NULL);
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::copyToAcceleratorAsync(accptr_t acc, IOBuffer &buffer,
    size_t bufferOff, size_t count, Mode &mode, cl_command_queue stream)
{
    trace::EnterCurrentFunction();
    uint8_t *host = buffer.addr() + bufferOff;
    TRACE(LOCAL,"Async copy to accelerator: %p ("FMT_SIZE")", host, count);

    buffer.toAccelerator(mode, stream);
    cl_int ret = clEnqueueWriteBuffer(stream, acc.base_, CL_FALSE,
        acc.offset_, count, host, 0, NULL, NULL);
    buffer.started();
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy to host: %p ("FMT_SIZE")", host, size);
    cl_int ret = clEnqueueReadBuffer(cmd_.front(), acc.base_,
        CL_TRUE, acc.offset_, size, host, 0, NULL, NULL);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyToHostAsync(IOBuffer &buffer, size_t bufferOff,
    const accptr_t acc, size_t count, Mode &mode, cl_command_queue stream)
{
    trace::EnterCurrentFunction();
    uint8_t *host = buffer.addr() + bufferOff;
    TRACE(LOCAL,"Async copy to host: %p ("FMT_SIZE")", host, count);
    buffer.toHost(mode, stream);
    cl_int ret = clEnqueueReadBuffer(stream, acc.base_, CL_FALSE,
        acc.offset_, count, host, 0, NULL, NULL);
    buffer.started();
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy accelerator-accelerator ("FMT_SIZE")", size);
    // TODO: This is a very inefficient implementation. We might consider
    // using a kernel for this task
    void *tmp = ::malloc(size);
    cl_int ret = clEnqueueReadBuffer(cmd_.front(), src.base_, CL_TRUE,
        src.offset_, size, tmp, 0, NULL, NULL);
    if(ret == CL_SUCCESS)
        ret = clEnqueueWriteBuffer(cmd_.front(), dst.base_, CL_TRUE,
            dst.offset_, size, tmp, 0, NULL, NULL);
    ::free(tmp);
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::memset(accptr_t addr, int c, size_t size)
{
    trace::EnterCurrentFunction();
    // TODO: This is a very inefficient implementation. We might consider
    // using a kernel for this task
    void *tmp = ::malloc(size);
    memset(tmp, c, size);
    cl_int ret = clEnqueueWriteBuffer(cmd_.front() , addr.base_,
        CL_TRUE, addr.offset_, size, tmp, 0, NULL, NULL);
    ::free(tmp);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::sync()
{
    trace::EnterCurrentFunction();
    cl_int ret = cmd_.sync();
    trace::ExitCurrentFunction();
    return error(ret);
}

cl_command_queue Accelerator::createCLstream()
{
    trace::EnterCurrentFunction();
    cl_command_queue stream;
    cl_int error;
    stream = clCreateCommandQueue(ctx_, device_, 0, &error);
    CFATAL(error == CL_SUCCESS, "Unable to create OpenCL stream");
    cmd_.add(stream);
    trace::ExitCurrentFunction();
    return stream;
}

void Accelerator::destroyCLstream(cl_command_queue stream)
{
    trace::EnterCurrentFunction();
    cl_int ret = clReleaseCommandQueue(stream);
    CFATAL(ret == CL_SUCCESS, "Unable to destroy OpenCL stream");
    cmd_.remove(stream);
    trace::ExitCurrentFunction();
}


gmacError_t Accelerator::syncCLstream(cl_command_queue stream)
{
    trace::EnterCurrentFunction();
    cl_int ret = clFinish(stream);
    trace::ExitCurrentFunction();
    return error(ret);
}

cl_int Accelerator::queryCLevent(cl_event event)
{
    trace::EnterCurrentFunction();
    cl_int ret;
    clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
        sizeof(cl_int), &ret, NULL);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Accelerator::syncCLevent(cl_event event)
{
    trace::EnterCurrentFunction();
    cl_int ret = clWaitForEvents(1, &event);
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::hostAlloc(hostptr_t *addr, size_t size)
{
    trace::EnterCurrentFunction();
    cl_int ret = CL_SUCCESS;
    cl_mem acc = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        size, NULL, &ret);
    if(ret == CL_SUCCESS) {
        // OpenCL works in the opposite way to CUDA, so we need to keep the
        // translation in map_
        *addr = (hostptr_t)clEnqueueMapBuffer(cmd_.front(), acc, CL_FALSE,
            CL_MAP_READ | CL_MAP_WRITE, 0, 0, 0, NULL, NULL, &ret);
        map_.insert(*addr, acc);
    }
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::hostFree(hostptr_t addr)
{
    trace::EnterCurrentFunction();
    cl_int ret = CL_SUCCESS;
    cl_mem device = map_.translate(addr);
    if(device != cl_mem(NULL)) {
        ret = clEnqueueUnmapMemObject(cmd_.front(), device, addr, 0, NULL, NULL);
        if(ret == CL_SUCCESS) ret = clReleaseMemObject(device);
    }
    trace::ExitCurrentFunction();
    return error(ret);
}


accptr_t Accelerator::hostMap(const hostptr_t addr)
{
    trace::EnterCurrentFunction();
    cl_mem device = map_.translate(addr);
    trace::ExitCurrentFunction();
    return accptr_t(device, 0);
}


void Accelerator::memInfo(size_t &free, size_t &total) const
{
    cl_int ret = CL_SUCCESS;
    cl_ulong value = 0;
    ret = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(value), &value, NULL);
    CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);
    total = value;

    // TODO: This is actually wrong, but OpenCL do not let us know the
    // amount of free memory in the accelerator
    ret = clGetDeviceInfo(device_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        sizeof(value), &value, NULL);
    CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);
    free = value;
}

}}
