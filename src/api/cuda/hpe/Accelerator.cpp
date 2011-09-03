#include "Accelerator.h"
#include "Mode.h"

#include "core/Process.h"

namespace __impl { namespace cuda { namespace hpe {

#ifdef USE_MULTI_CONTEXT
util::Private<CUcontext> Accelerator::Ctx_;
#endif

void
Switch::in(Mode &mode)
{
    mode.getAccelerator().pushContext();
}

void
Switch::out(Mode &mode)
{
    mode.getAccelerator().popContext();
}

Accelerator::Accelerator(int n, CUdevice device) :
    gmac::core::hpe::Accelerator(n), device_(device)
#ifndef USE_MULTI_CONTEXT
#ifdef USE_VM
    , lastMode_(NULL)
#endif
    , ctx_(NULL)
#endif
{
#if CUDA_VERSION > 3010
    size_t size = 0;
#else
    unsigned int size = 0;
#endif
    TRACE(GLOBAL, "Creating CUDA accelerator %d", device_);
    CUresult ret = cuDeviceTotalMem(&size, device_);
    CFATAL(ret == CUDA_SUCCESS, "Unable to initialize CUDA device %d", ret);
    ret = cuDeviceComputeCapability(&major_, &minor_, device_);
    CFATAL(ret == CUDA_SUCCESS, "Unable to initialize CUDA device %d", ret);

    int val;

#if CUDA_VERSION > 3000
    ret = cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, n);
    CFATAL(ret == CUDA_SUCCESS, "Unable to get attribute %d", ret);
    busId_ = val;
    ret = cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, n);
    CFATAL(ret == CUDA_SUCCESS, "Unable to get attribute %d", ret);
    busAccId_ = val;
#endif
    ret = cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_INTEGRATED, n);
    CFATAL(ret == CUDA_SUCCESS, "Unable to get attribute %d", ret);
    integrated_ = (val != 0);

#ifndef USE_MULTI_CONTEXT
    CUcontext tmp;
    unsigned int flags = 0;
#if CUDA_VERSION >= 2020
    if(major_ >= 2 || (major_ == 1 && minor_ >= 1)) flags |= CU_CTX_MAP_HOST;
#else
    TRACE(LOCAL,"Host mapped memory not supported by the HW");
#endif

    ret = cuCtxCreate(&ctx_, flags, device_);
    CFATAL(ret == CUDA_SUCCESS, "Unable to create CUDA context %d", ret);

#if defined(USE_TRACE)
    ret = cuEventCreate(&start_, CU_EVENT_DEFAULT);
    CFATAL(ret == CUDA_SUCCESS);
    ret = cuEventCreate(&end_, CU_EVENT_DEFAULT);
    CFATAL(ret == CUDA_SUCCESS);
#endif

    ret = cuCtxPopCurrent(&tmp);
    CFATAL(ret == CUDA_SUCCESS, "Error setting up a new context %d", ret);
#else
#endif

}

Accelerator::~Accelerator()
{
#ifndef USE_MULTI_CONTEXT
#ifdef CALL_CUDA_ON_DESTRUCTION
    pushContext();
#endif // CALL_CUDA_ON_DESTRUCTION
#ifdef CALL_CUDA_ON_DESTRUCTION
    popContext();
    CUresult ret = cuCtxDestroy(ctx_);
    ASSERTION(ret == CUDA_SUCCESS);

#endif // CALL_CUDA_ON_DESTRUCTION 
#endif // USE_MULTI_CONTEXT

#if defined(USE_TRACE)
#ifdef CALL_CUDA_ON_DESTRUCTION
    cuEventDestroy(start_);
    cuEventDestroy(end_);
#endif
#endif // USE_TRACE

}

void
Accelerator::init()
{
#ifdef USE_MULTI_CONTEXT
    util::Private<CUcontext>::init(Ctx_);
#endif
}

core::hpe::Mode *
Accelerator::createMode(core::hpe::Process &proc)
{
    trace::EnterCurrentFunction();
    core::hpe::Mode *mode = ModeFactory::create(proc, *this);
    if (mode != NULL) {
        registerMode(*mode);
    }
    trace::ExitCurrentFunction();

    TRACE(LOCAL,"Creating Execution Mode %p to Accelerator", mode);
    return mode;
}

#ifdef USE_MULTI_CONTEXT
CUcontext
Accelerator::createCUcontext()
{
    trace::EnterCurrentFunction();
    CUcontext ctx, tmp;
    unsigned int flags = 0;
#if CUDA_VERSION >= 2020
    if(major_ >= 2 || (major_ == 1 && minor_ >= 1)) flags |= CU_CTX_MAP_HOST;
#else
    TRACE(LOCAL,"Host mapped memory not supported by the HW");
#endif
    CUresult ret = cuCtxCreate(&ctx, flags, device_);
    if(ret != CUDA_SUCCESS)
        FATAL("Unable to create CUDA context %d", ret);
    ret = cuCtxPopCurrent(&tmp);
    ASSERTION(ret == CUDA_SUCCESS);
    trace::ExitCurrentFunction();
    return ctx;
}

void
Accelerator::destroyCUcontext(CUcontext ctx)
{
    trace::EnterCurrentFunction();
    CFATAL(cuCtxDestroy(ctx) == CUDA_SUCCESS, "Error destroying CUDA context");
    trace::ExitCurrentFunction();
}

#endif

#ifdef USE_MULTI_CONTEXT
ModuleVector
Accelerator::createModules()
{
    trace::EnterCurrentFunction();
    pushContext();
    ModuleVector modules = ModuleDescriptor::createModules();
    popContext();
    trace::ExitCurrentFunction();
    return modules;
}

void
Accelerator::destroyModules(ModuleVector &modules)
{
    trace::EnterCurrentFunction();
    pushContext();
    modules.clear();
    popContext();
    trace::ExitCurrentFunction();
}

#else
ModuleVector &
Accelerator::createModules()
{
    trace::EnterCurrentFunction();
    if(modules_.empty()) {
        pushContext();
        modules_ = ModuleDescriptor::createModules();
        popContext();
    }
    trace::ExitCurrentFunction();
    return modules_;
}
#endif

gmacError_t
Accelerator::map(accptr_t &dst, hostptr_t src, size_t count, unsigned align)
{
    trace::EnterCurrentFunction();
    dst = accptr_t(0);
#if CUDA_VERSION >= 3020
    size_t gpuSize = count;
#else
    unsigned gpuSize = unsigned(count);
#endif
    if(align > 1) {
        gpuSize += align;
    }
    CUdeviceptr ptr = 0;
    pushContext();
    CUresult ret = cuMemAlloc(&ptr, gpuSize);
    popContext();
    if(ret != CUDA_SUCCESS) {
        trace::ExitCurrentFunction();
        return error(ret);
    }

    CUdeviceptr gpuPtr = ptr;
    if(gpuPtr % align) {
        gpuPtr += align - (gpuPtr % align);
    }
    dst = gpuPtr;

#ifndef USE_MULTI_CONTEXT
    dst.pasId_ = id_;
#endif

    allocations_.insert(src, dst, count);

    alignMap_.lockWrite();

    alignMap_.insert(AlignmentMap::value_type(gpuPtr, ptr));
    alignMap_.unlock();
    TRACE(LOCAL,"Allocating device memory: %p (originally %p) - "FMT_SIZE" (originally "FMT_SIZE") bytes (alignment %u)", dst.get(), ptr, gpuSize, count, align);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t
Accelerator::unmap(hostptr_t host, size_t count)
{
    trace::EnterCurrentFunction();
    ASSERTION(host != NULL);

    accptr_t addr(0);
    size_t s;

    bool hasMapping = allocations_.find(host, addr, s);
    ASSERTION(hasMapping == true);
    ASSERTION(s == count);
    allocations_.erase(host, count);

    TRACE(LOCAL, "Releasing accelerator memory @ %p", addr.get());

    AlignmentMap::iterator i;
    alignMap_.lockWrite();
    i = alignMap_.find(addr);
    if (i == alignMap_.end()) {
        alignMap_.unlock();
        trace::ExitCurrentFunction();
        return gmacErrorInvalidValue;
    }
    CUdeviceptr device = i->second;
    alignMap_.erase(i);
    alignMap_.unlock();
    pushContext();
    trace::SetThreadState(trace::Wait);
    CUresult ret = cuMemFree(device);
    trace::SetThreadState(trace::Running);
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t
Accelerator::memset(accptr_t addr, int c, size_t count, stream_t /*stream*/)
{
    trace::EnterCurrentFunction();
    CUresult ret = CUDA_SUCCESS;
    pushContext();
    if(count % 4 == 0) {
        int seed = c | (c << 8) | (c << 16) | (c << 24);
#if CUDA_VERSION >= 3020
        ret = cuMemsetD32(addr, seed, count / 4);
#else
        ret = cuMemsetD32(addr, seed, unsigned(count / 4));
#endif
    } else if(count % 2) {
        short s = (short) c & 0xffff;
        short seed = s | (s << 8);
#if CUDA_VERSION >= 3020
        ret = cuMemsetD16(addr, seed, count / 2);
#else
        ret = cuMemsetD16(addr, seed, unsigned(count / 2));
#endif
    } else {
#if CUDA_VERSION >= 3020
        ret = cuMemsetD8(addr, (uint8_t)(c & 0xff), count);
#else
        ret = cuMemsetD8(addr, (uint8_t)(c & 0xff), unsigned(count));
#endif
    }
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::sync()
{
    trace::EnterCurrentFunction();
    pushContext();
    CUresult ret = cuCtxSynchronize();
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::hostAlloc(hostptr_t &addr, size_t size, GmacProtection prot)
{
    trace::EnterCurrentFunction();
#if CUDA_VERSION >= 2020
    unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
    if (prot == GMAC_PROT_WRITE) {
        flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    pushContext();
    CUresult ret = cuMemHostAlloc((void **) &addr, size, flags);
    popContext();
#else
        CUresult ret = CUDA_ERROR_OUT_OF_MEMORY;
#endif
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::hostFree(hostptr_t addr)
{
    trace::EnterCurrentFunction();
#if CUDA_VERSION >= 2020
    pushContext();
    CUresult r = cuMemFreeHost(addr);
    popContext();
#else
    CUresult r = CUDA_ERROR_OUT_OF_MEMORY;
#endif
    trace::ExitCurrentFunction();
    return error(r);
}

accptr_t Accelerator::hostMap(const hostptr_t addr)
{
    trace::EnterCurrentFunction();
#if CUDA_VERSION >= 2020
    CUdeviceptr device;
    pushContext();
    CUresult ret = cuMemHostGetDevicePointer(&device, addr, 0);
    popContext();
#else
    CUresult ret = CUDA_ERROR_OUT_OF_MEMORY;
#endif
    if(ret != CUDA_SUCCESS) device = 0;
    trace::ExitCurrentFunction();
    return accptr_t(device);
}

void Accelerator::getMemInfo(size_t &free, size_t &total) const
{
    pushContext();

#if CUDA_VERSION > 3010
    CUresult ret = cuMemGetInfo(&free, &total);
#else
    CUresult ret = cuMemGetInfo((unsigned int *) &free, (unsigned int *) &total);
#endif
    CFATAL(ret == CUDA_SUCCESS, "Error getting memory info");
    popContext();
}

gmacError_t Accelerator::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, core::hpe::Mode &mode)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy to accelerator: %p -> %p ("FMT_SIZE")", host, acc.get(), size);
    trace::SetThreadState(trace::Wait);
    pushContext();
    CUresult ret = CUDA_SUCCESS;
#if USE_TRACE
    ret = cuEventRecord(start_, 0);
    ASSERTION(ret == CUDA_SUCCESS);
    trace::SetThreadState(trace::Wait);
#endif
#if CUDA_VERSION >= 3020
    ret = cuMemcpyHtoD(acc, host, size);
#else
    ret = cuMemcpyHtoD(CUdeviceptr(acc), host, unsigned(size));
#endif
#if USE_TRACE
    ret = cuEventRecord(end_, 0);
    ret = cuEventSynchronize(end_);
    trace::SetThreadState(trace::Running);
    DataCommToAccelerator(dynamic_cast<Mode &>(mode), start_, end_, size);
#endif
    popContext();
    trace::SetThreadState(trace::Running);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyToAcceleratorAsync(accptr_t acc, core::IOBuffer &_buffer, size_t bufferOff, size_t count, core::hpe::Mode &mode, CUstream stream)
{
    IOBuffer &buffer = dynamic_cast<IOBuffer &>(_buffer);
    trace::EnterCurrentFunction();
    uint8_t *host = buffer.addr() + bufferOff;
    TRACE(LOCAL,"Async copy to accelerator: %p -> %p ("FMT_SIZE")", host, acc.get(), count);
    pushContext();

    buffer.toAccelerator(dynamic_cast<cuda::Mode &>(mode), stream);
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyHtoDAsync(acc, host, count, stream);
#else
    CUresult ret = cuMemcpyHtoDAsync(CUdeviceptr(acc), host, unsigned(count), stream);
#endif
    buffer.started(count);
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyToHost(hostptr_t host, const accptr_t acc, size_t size, core::hpe::Mode &mode)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy to host: %p -> %p ("FMT_SIZE")", acc.get(), host, size);
    trace::SetThreadState(trace::Wait);
    pushContext();
    CUresult ret;
#if USE_TRACE
    ret = cuEventRecord(start_, 0);
    ASSERTION(ret == CUDA_SUCCESS);
    trace::SetThreadState(trace::Wait);
#endif

#if CUDA_VERSION >= 3020
    ret = cuMemcpyDtoH(host, acc, size);
#else
    ret = cuMemcpyDtoH(host, acc, unsigned(size));
#endif
#if USE_TRACE
    ret = cuEventRecord(end_, 0);
    ret = cuEventSynchronize(end_);
    trace::SetThreadState(trace::Running);
    DataCommToHost(dynamic_cast<Mode &>(mode), start_, end_, size);
#endif

    popContext();
    trace::SetThreadState(trace::Running);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyToHostAsync(core::IOBuffer &_buffer, size_t bufferOff, const accptr_t acc, size_t count, core::hpe::Mode &mode, CUstream stream)
{
    IOBuffer &buffer = dynamic_cast<IOBuffer &>(_buffer);
    trace::EnterCurrentFunction();
    uint8_t *host = buffer.addr() + bufferOff;
    TRACE(LOCAL,"Async copy to host: %p -> %p ("FMT_SIZE")", acc.get(), host, count);
    pushContext();
    buffer.toHost(dynamic_cast<cuda::Mode &>(mode), stream);
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyDtoHAsync(host, acc, count, stream);
#else
    CUresult ret = cuMemcpyDtoHAsync(host, acc, unsigned(count), stream);
#endif
    buffer.started(count);
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyAccelerator(accptr_t dst, const accptr_t src, size_t size, stream_t stream)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy accelerator-accelerator: %p -> %p ("FMT_SIZE")", src.get(), dst.get(), size);
    pushContext();
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyDtoDAsync(dst, src, size, stream);
#else
    CUresult ret = cuMemcpyDtoDAsync(dst, src, unsigned(size), stream);
#endif
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::execute(KernelLaunch &launch)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Executing KernelLaunch");
    pushContext();
    gmacError_t ret = launch.execute();
    popContext();
    trace::ExitCurrentFunction();
    return ret;
}

#if CUDA_VERSION >= 4000 
gmacError_t Accelerator::registerMem(hostptr_t ptr, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Executing KernelLaunch");
    CUresult ret = cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_PORTABLE);
    CFATAL(ret == CUDA_SUCCESS);
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::unregisterMem(hostptr_t ptr)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Executing KernelLaunch");
    pushContext();
    CUresult ret = cuMemHostUnregister(ptr);
    CFATAL(ret == CUDA_SUCCESS);
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}
#endif

CUstream Accelerator::createCUstream()
{
    trace::EnterCurrentFunction();
    CUstream stream;
    pushContext();
    CUresult ret = cuStreamCreate(&stream, 0);
    popContext();
    CFATAL(ret == CUDA_SUCCESS, "Unable to create CUDA stream");
    trace::ExitCurrentFunction();
    return stream;
}

void Accelerator::destroyCUstream(CUstream stream)
{
    trace::EnterCurrentFunction();
    pushContext();
    CUresult ret = cuStreamDestroy(stream);
    popContext();
    CFATAL(ret == CUDA_SUCCESS, "Unable to destroy CUDA stream");
    trace::ExitCurrentFunction();
}


}}}
