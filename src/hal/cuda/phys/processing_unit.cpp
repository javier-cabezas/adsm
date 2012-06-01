#include "trace/logger.h"

#include "hal/types.h"
#include "aspace.h"
#include "platform.h"
#include "processing_unit.h"

namespace __impl { namespace hal { namespace cuda { namespace phys {

processing_unit::processing_unit(platform &plat, aspace &as, virt::scheduler &sched, CUdevice cudaDevice) :
    parent(plat, parent::PUNIT_TYPE_GPU, as, sched),
    lock("processing_unit"),
    cudaDevice_(cudaDevice),
    isInfoInitialized_(false)
{
    CUresult ret;
    int val;

    TRACE(GLOBAL, "Creating CUDA accelerator %d", cudaDevice_);
    ret = cuDeviceComputeCapability(&major_, &minor_, cudaDevice_);
    CFATAL(ret == CUDA_SUCCESS, "Unable to initialize CUDA device %d", ret);
    ret = cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_INTEGRATED, cudaDevice_);
    CFATAL(ret == CUDA_SUCCESS, "Unable to get attribute %d", ret);
    integrated_ = (val != 0);
}

hal_stream *
processing_unit::create_stream(virt::hal_aspace &_as)
{
    virt::aspace &as = reinterpret_cast<virt::aspace &>(_as);
    as.set(); 

    CUstream s;
    CUresult ret = cuStreamCreate(&s, 0);
    CFATAL(ret == CUDA_SUCCESS, "Unable to create CUDA stream");

    return new stream(as, s);
}

hal::error
processing_unit::destroy_stream(hal_stream &_s)
{
    stream &s = reinterpret_cast<stream &>(_s);
    CUresult ret = cuStreamDestroy(s());
    delete &s;

    return error_to_hal(ret);
}

CUdevice
processing_unit::get_cuda_id() const
{
    return cudaDevice_;
}

int
processing_unit::get_major() const
{
    return major_;
}

int
processing_unit::get_minor() const
{
    return minor_;
}

size_t
processing_unit::get_total_memory() const
{
    size_t total, dummy;
    CUresult ret = cuMemGetInfo(&dummy, &total);
    CFATAL(ret == CUDA_SUCCESS, "Error getting processing unit memory size: %d", ret);
    return total;
}

size_t
processing_unit::get_free_memory() const
{
    size_t free, dummy;
    CUresult ret = cuMemGetInfo(&free, &dummy);
    CFATAL(ret == CUDA_SUCCESS, "Error getting processing unit memory size: %d", ret);
    return free;
}

hal::error
processing_unit::get_info(GmacDeviceInfo &info)
{
    lock::lock();
    if (!isInfoInitialized_) {
        static const size_t MaxAcceleratorNameLength = 128;
        char deviceName[MaxAcceleratorNameLength];
        CUresult res = cuDeviceGetName(deviceName, MaxAcceleratorNameLength, cudaDevice_);
        ASSERTION(res == CUDA_SUCCESS);

        info_.deviceName = deviceName;
        info_.vendorName = "NVIDIA Corporation";
        info_.deviceType = GMAC_PUNIT_TYPE_GPU;
        info_.vendorId = 1;
        info_.isAvailable = 1;
        
        int val;
        res = cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudaDevice_);
        ASSERTION(res == CUDA_SUCCESS);
        info_.computeUnits = val;

        CUdevprop deviceProperties;

        res = cuDeviceGetProperties(&deviceProperties, cudaDevice_);
        ASSERTION(res == CUDA_SUCCESS);

        info_.maxDimensions = 3;

        size_t *maxSizes = new size_t[3];
        maxSizes[0] = deviceProperties.maxThreadsDim[0];
        maxSizes[1] = deviceProperties.maxThreadsDim[1];
        maxSizes[2] = deviceProperties.maxThreadsDim[2];

        info_.maxSizes = maxSizes;
        info_.maxWorkGroupSize = deviceProperties.maxThreadsPerBlock;
        info_.globalMemSize = get_total_memory();
        info_.localMemSize  = deviceProperties.sharedMemPerBlock;
        info_.cacheMemSize  = 0;

        int cudaVersion = 0;
        res = cuDriverGetVersion(&cudaVersion);
        if(res == CUDA_SUCCESS) info_.driverMajor = cudaVersion;
        else info_.driverMajor = 0;
        info_.driverMinor = info_.driverRev = 0;

        isInfoInitialized_ = true;
    }
    lock::unlock();

    info = info_;
    return hal::error::HAL_SUCCESS;
}

#if 0
cpu::cpu(platform &plat, coherence_domain &coherenceDomain) :
    parent(parent::DEVICE_TYPE_CPU, plat, coherenceDomain)
{
}

aspace *
cpu::create_context(const set_siblings &siblings, hal::error &err)
{
    //return new context_cpu(siblings);
    return NULL;
}

hal::error
cpu::destroy_context(aspace &context)
{
    delete &context;

    return hal::error::HAL_SUCCESS;
}

stream_t *
cpu::create_stream(aspace &context)
{
    return NULL;
}

hal::error
cpu::destroy_stream(stream_t &stream)
{
    return hal::error::HAL_SUCCESS;
}

size_t
cpu::get_total_memory() const
{
    FATAL("Not implemented");
    return 0;
}

size_t
cpu::get_free_memory() const
{
    FATAL("Not implemented");
    return 0;
}

bool
cpu::has_direct_copy(const device &dev) const
{
    return true;
}

hal::error
cpu::get_info(GmacDeviceInfo &info)
{
    return hal::error::HAL_SUCCESS;
}
#endif

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
