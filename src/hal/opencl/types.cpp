#include "config/common.h"

#include "hal/opencl/helper/opencl_helper.h"

#include "coherence_domain.h"
#include "device.h"
#include "module.h"

#define __GMAC_ERROR(r, err) case r: error = err; break

namespace __impl { namespace hal {

gmacError_t
init_platform()
{
    static bool initialized = false;
    gmacError_t ret = gmacSuccess;

    if (initialized == false) {
        initialized = true;
    } else {
        FATAL("Double HAL platform initialization");
    }

    return ret;
}

opencl::map_context_repository Modules_("map_context_repository");

std::list<opencl::device *>
init_devices()
{
    static bool initialized = false;

    if (initialized == false) {
        initialized = true;
    } else {
        FATAL("Double HAL device initialization");
    }

    std::list<opencl::device *> devices;

    TRACE(GLOBAL, "Initializing OpenCL API");
    cl_uint platformSize = 0;
    cl_int ret = CL_SUCCESS;
    ret = clGetPlatformIDs(0, NULL, &platformSize);
    CFATAL(ret == CL_SUCCESS);
    if(platformSize == 0) return devices;   
    cl_platform_id * platforms = new cl_platform_id[platformSize];
    ret = clGetPlatformIDs(platformSize, platforms, NULL);
    CFATAL(ret == CL_SUCCESS);
    MESSAGE("%d OpenCL platforms found", platformSize);

    unsigned n = 0;
    for (unsigned i = 0; i < platformSize; i++) {
        MESSAGE("Platform [%u/%u]: %s", i + 1, platformSize, opencl::helper::get_platform_name(platforms[i]).c_str());
        cl_uint deviceSize = 0;
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU,
                             0, NULL, &deviceSize);
        ASSERTION(ret == CL_SUCCESS);
	    if(deviceSize == 0) continue;
        cl_device_id *devices = new cl_device_id[deviceSize];
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU,
                             deviceSize, devices, NULL);
        ASSERTION(ret == CL_SUCCESS);
        MESSAGE("... found %u OpenCL devices", deviceSize, i);

        opencl::helper::opencl_version clVersion = opencl::helper::get_opencl_version(platforms[i]);

        cl_context ctx;

        if (deviceSize > 0) {
            cl_context_properties prop[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0 };

            ctx = clCreateContext(prop, deviceSize, devices, NULL, NULL, &ret);
            CFATAL(ret == CL_SUCCESS, "Unable to create OpenCL context %d", ret);
        }

        for (unsigned j = 0; j < deviceSize; j++) {
            MESSAGE("Device [%u/%u]: %s", j + 1, deviceSize, opencl::helper::get_device_name(devices[j]).c_str());

            // Let's assume that this is not important; TODO: actually deal with this case
            //CFATAL(util::getDeviceVendor(devices[j]) == util::getPlatformVendor(platforms[i]), "Not handled case");
            opencl::device *device = NULL;

            switch (opencl::helper::get_platform(platforms[i])) {
                case opencl::helper::PLATFORM_AMD:
                    if (opencl::helper::is_device_amd_fusion(devices[j])) {
                        device = new opencl::device(platforms[i], devices[j], *new opencl::coherence_domain());
                    } else {
                        device = new opencl::device(platforms[i], devices[j], *new opencl::coherence_domain());
                    }
                    break;
                case opencl::helper::PLATFORM_APPLE:
                case opencl::helper::PLATFORM_NVIDIA:
                    device = new opencl::device(platforms[i], devices[j], *new opencl::coherence_domain());
                    break;
                case opencl::helper::PLATFORM_INTEL:
                case opencl::helper::PLATFORM_UNKNOWN:
                    FATAL("Platform not supported\n");
            }
            devices.push_back(device);
        }
        if (deviceSize > 0) {
            ret = clReleaseContext(ctx);
            CFATAL(ret == CL_SUCCESS, "Unable to release OpenCL context after accelerator initialization");
        }
        delete[] devices;
    }
    delete[] platforms;
    initialized = true;

    opencl::compile_embedded_code(devices);

    return devices;
}

}}

namespace __impl { namespace hal { namespace cuda { 

gmacError_t error(CUresult err)
{
    gmacError_t error = gmacSuccess;
    switch(err) {
        __GMAC_ERROR(CL_SUCCESS, gmacSuccess);
        __GMAC_ERROR(CL_DEVICE_NOT_FOUND, gmacErrorNoAccelerator);
        __GMAC_ERROR(CL_DEVICE_NOT_AVAILABLE, gmacErrorInvalidAccelerator);
        __GMAC_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CL_OUT_OF_HOST_MEMORY, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CL_OUT_OF_RESOURCES, gmacErrorMemoryAllocation);

        default: error = gmacErrorUnknown;
    }
    return error;
}

_event_t *
context_t::get_new_event(bool async,_event_t::type t)
{
    _event_t *ret = queueEvents_.pop();
    if (ret == NULL) {
        ret = new _event_t(async, t, *this);
    } else {
        ret->reset(async, t);
    }

    return ret;
}

void
context_t::dispose_event(_event_t &event)
{
    queueEvents_.push(event);
}

context_t::context_t(CUcontext ctx, device &dev) :
    Parent(ctx, dev)
{
    TRACE(LOCAL, "Creating context: %p", (*this)());
}

accptr_t
context_t::alloc(size_t count, gmacError_t &err)
{
    CUdeviceptr devPtr = 0;
    CUresult res = cuMemAlloc(&devPtr, count);

    err = error(res);

    return accptr_t(devPtr);
}

buffer_t *
context_t::alloc_buffer(size_t count, GmacProtection hint, gmacError_t &err)
{
    // TODO: add a parater to specify accesibility of the buffer from the device
    unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
    if (hint == GMAC_PROT_WRITE) {
        flags += CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    void *addr;
    CUresult res = cuMemHostAlloc(&addr, count, flags);
    err = error(res);

    return new buffer_t(hostptr_t(addr), *this);
}

gmacError_t
context_t::free(accptr_t acc)
{
    CUresult ret = cuMemFree(acc.get());

    return error(ret);
}

gmacError_t
context_t::free_buffer(buffer_t &buffer)
{
    CUresult ret = cuMemFreeHost(buffer.get_addr());

    return error(ret);
}

event_t 
context_t::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "H (%p) -> D copy ("FMT_SIZE" bytes) on stream: %p", src, count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    CUresult res;

    ret.begin(stream);
    res = cuMemcpyHtoD(dst.get(), src, count);
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p", dst, count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    CUresult res;

    ret.begin(stream);
    res = cuMemcpyDtoH(dst, src.get(), count);
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> D copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    CUresult res;

    ret.begin(stream);
    res = cuMemcpyDtoD(dst.get(), src.get(), count);
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, src, off, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, src, off, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "H (%p) -> D copy_async ("FMT_SIZE" bytes) on stream: %p", src.get_addr() + off, count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    CUresult res;

    ret.begin(stream);
    res = cuMemcpyHtoDAsync(dst.get(), src.get_addr() + off, count, stream());
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, off, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, off, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> H (%p) copy_async ("FMT_SIZE" bytes) on stream: %p", dst.get_addr(), count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    CUresult res;

    ret.begin(stream);
    res = cuMemcpyDtoHAsync(dst.get_addr() + off, src.get(), count, stream());
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, src, count, stream, err);
    }

    return ret;
}

event_t
context_t::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    return copy_async(0, NULL, dst, src, count, stream, err);
}

event_t 
context_t::copy_async(unsigned nevents, cl_event *events, accptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> D copy_async ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    cl_int res;

    ret.begin(stream);
    res = clEnqueueCopyBuffer(stream(), src.get(),    dst.get(),
                                        src.offset(), dst.offset(), count,
                                        nevents, events,
                                        &ret());
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::memset(accptr_t dst, int c, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = memset(dst, c, count, stream, err);
    }

    return ret;
}

event_t 
context_t::memset(accptr_t dst, int c, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = memset(dst, c, count, stream, err);
    }

    return ret;
}

event_t 
context_t::memset(accptr_t dst, int c, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "memset ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    ret.begin(stream);
    CUresult res = cuMemsetD8(dst.get(), (unsigned char)c, count);
    ret.end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::memset_async(accptr_t dst, int c, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = memset_async(dst, c, count, stream, err);
    }

    return ret;
}

event_t 
context_t::memset_async(accptr_t dst, int c, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = memset_async(dst, c, count, stream, err);
    }

    return ret;
}

event_t 
context_t::memset_async(accptr_t dst, int c, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "memset_async ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    ret.begin(stream);
    CUresult res = cuMemsetD8Async(dst.get(), (unsigned char)c, count, stream());
    ret.end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

const code_repository &
context_t::get_code_repository()
{
    code_repository *repository;
    map_context_repository::iterator it = Modules_.find(this);
    if (it == Modules_.end()) {
        repository = module_descriptor::create_modules();
        Modules_.insert(map_context_repository::value_type(this, repository));
    } else {
        repository = it->second;
    }

    return *repository;
}

stream_t::stream_t(CUstream stream, context_t &context) :
    Parent(stream, context)
{
    TRACE(LOCAL, "Creating stream: %p", (*this)());
}

gmacError_t
stream_t::sync()
{
    TRACE(LOCAL, "Waiting for stream: %p", (*this)());
    cl_int ret = clFinish((*this)());

    return error(ret);
}

void
_event_t::reset(bool async, type t)
{
    isAsynchronous_ = async;
    type_ = t;
    err_ = gmacSuccess;
    synced_ = false;
    state_ = None;
}

_event_t::state
_event_t::get_state()
{
    if (state_ != End) {
        get_stream().get_context().set();

        CUresult res = cuEventQuery(eventEnd_);

        if (res == CUDA_ERROR_NOT_READY) {
            state_ = Queued;
        } else if (res == CUDA_SUCCESS) {
            state_ = End;
        }
    }

    return state_;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
