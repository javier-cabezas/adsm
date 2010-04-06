#include "Context.h"

namespace gmac { namespace gpu {

Context::AddressMap Context::hostMem;
void * Context::FatBin;
#ifdef USE_VM
const char *Context::pageTableSymbol = "__pageTable";
#endif

void
Context::setup()
{
#ifdef USE_MULTI_CONTEXT
    _ctx = _gpu.createCUDAContext();
    enable();
#endif
}

void
Context::setupStreams()
{
    if (_gpu.async()) {
        CUresult ret;
        ret = cuStreamCreate(&streamLaunch, 0);
        CFATAL(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
        ret = cuStreamCreate(&streamToDevice, 0);
        CFATAL(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
        ret = cuStreamCreate(&streamToHost, 0);
        CFATAL(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
        ret = cuStreamCreate(&streamDevice, 0);
        CFATAL(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
    }

    if (_gpu.async()) {
        _bufferPageLockedSize = paramBufferPageLockedSize * paramPageSize;
#if CUDART_VERSION >= 2020
        CUresult ret = cuMemHostAlloc(&_bufferPageLocked, _bufferPageLockedSize, CU_MEMHOSTALLOC_PORTABLE);
#else
        CUresult ret = cuMemAllocHost(&_bufferPageLocked, _bufferPageLockedSize);
#endif
        if (ret == CUDA_SUCCESS) {
            TRACE("Using page locked memory: %zd\n", _bufferPageLockedSize);
        } else {
            FATAL("Error %d: when allocating page-locked memory", ret);
        }
    } else {
        _bufferPageLocked     = NULL;
        _bufferPageLockedSize = 0;
        TRACE("Not using page locked memory\n");
    }
}

Context::Context(Accelerator &gpu) :
    gmac::Context(gpu),
    _gpu(gpu),
    _call(dim3(0), dim3(0), 0, 0)
#ifdef USE_VM
    , pageTable(NULL)
#endif
#ifdef USE_MULTI_CONTEXT
    , mutex(paraver::ctxLocal)
#endif
    , _pendingKernel(false)
    , _pendingToDevice(false)
    , _pendingToHost(false)
{
	setup();

    lock();
    setupStreams();
    TRACE("Let's create modules");
    _modules = ModuleDescriptor::createModules();
    unlock();

    TRACE("New Accelerator context [%p]", this);
}

Context::~Context()
{
    TRACE("Remove Accelerator context [%p]", this);
    if (_gpu.async()) {
        cuStreamDestroy(streamLaunch);
        cuStreamDestroy(streamToDevice);
        cuStreamDestroy(streamToHost);
        cuStreamDestroy(streamDevice);
    }
    // CUDA might be deinitalized before executing this code
    // mutex->lock();
    // ASSERT(cuCtxDestroy(_ctx) == CUDA_SUCCESS);
}

gmacError_t
Context::malloc(void **addr, size_t size) {
    ASSERT(addr);
    zero(addr);
    lock();
    size += mm().pageTable().getPageSize();
    CUdeviceptr ptr = 0;
    CUresult ret = cuMemAlloc(&ptr, size);
    CUdeviceptr gpuPtr = ptr;
    if(gpuPtr % mm().pageTable().getPageSize()) {
        gpuPtr += mm().pageTable().getPageSize() -
            (gpuPtr % mm().pageTable().getPageSize());
    }
    *addr = (void *)gpuPtr;
    _alignMap.insert(AlignmentMap::value_type(gpuPtr, ptr));
    unlock();
    return error(ret);
}

gmacError_t
Context::mallocPageLocked(void **addr, size_t size)
{
    ASSERT(addr);
    zero(addr);
    CUresult ret = CUDA_SUCCESS;
    lock();
#if CUDART_VERSION >= 2020
    ret = cuMemHostAlloc(addr, size, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
#else
    ret = cuMemAllocHost(&_bufferPageLocked, _bufferPageLockedSize);
#endif
    unlock();
    return error(ret);
}


gmacError_t
Context::free(void *addr)
{
    ASSERT(addr);
    lock();
    AlignmentMap::const_iterator i;
    CUdeviceptr gpuPtr = gpuAddr(addr);
    i = _alignMap.find(gpuPtr);
    if (i == _alignMap.end()) return gmacErrorInvalidValue;
    CUresult ret = cuMemFree(i->second);
    unlock();
    return error(ret);
}

gmacError_t
Context::mapToDevice(void *host, void **device, size_t size)
{
    ASSERT(host   != NULL);
    ASSERT(device != NULL);
    ASSERT(size   > 0);
    Accelerator & acc = static_cast<Accelerator &>(_acc);
    CFATAL(acc.major() >= 2 ||
          (acc.major() == 1 && acc.minor() >= 1), "Map to device not supported by the HW");
    zero(device);
    CUresult ret = CUDA_SUCCESS;
#if CUDART_VERSION >= 2020
    lock();
    ret = cuMemHostGetDevicePointer((CUdeviceptr *)device, host, 0);
    hostMem.insert(AddressMap::value_type(host, *device));
    unlock();
#else
    FATAL("Map to device not supported by the CUDA version");
#endif
    return error(ret);
}


gmacError_t
Context::hostFree(void *addr)
{
    ASSERT(addr);
	lock();
	AddressMap::iterator i = hostMem.find(addr);
	ASSERT(i != hostMem.end());
	CUresult ret = cuMemFreeHost(i->second);
	hostMem.erase(i);
	unlock();
	return error(ret);
}

gmacError_t
Context::copyToDevice(void *dev, const void *host, size_t size)
{
    enterFunction(FuncAccHostDevice);
    gmac::Context *ctx = gmac::Context::current();

    TRACE("Copy to device: %p to %p", host, dev);
    CUresult ret = CUDA_SUCCESS;

    if (_gpu.async()) {
        lock();
        if (_pendingToDevice) {
            if ((ret = cuStreamQuery(streamToDevice)) != CUDA_SUCCESS) {
                ASSERT(ret == CUDA_ERROR_NOT_READY);
                ret = cuStreamSynchronize(streamToDevice);
            }
            popEventState(paraver::Accelerator, 0x10000000 + _id);
        } 
        _pendingToDevice = false;
        unlock();

        size_t bufferSize = ctx->bufferPageLockedSize();
        void * tmp        = ctx->bufferPageLocked();
        if (size > _bufferPageLockedSize) {
            size_t left = size;
            off_t  off  = 0;
            while (left != 0) {
                size_t bytes = left < bufferSize? left: bufferSize;
                memcpy(tmp, ((char *) host) + off, bytes);
                lock();
                pushEventState(IOWrite, paraver::Accelerator, 0x10000000 + _id, AcceleratorIO);
                ret = cuMemcpyHtoDAsync(gpuAddr(((char *) dev) + off), tmp, bytes, streamToDevice);
                CBREAK(ret == CUDA_SUCCESS, unlock());
                ret = cuStreamSynchronize(streamToDevice);
                popEventState(paraver::Accelerator, 0x10000000 + _id);
                unlock();
                CBREAK(ret == CUDA_SUCCESS);

                left -= bytes;
                off  += bytes;
            }
        } else {
            memcpy(tmp, host, size);
            lock();
            pushEventState(IOWrite, paraver::Accelerator, 0x10000000 + _id, AcceleratorIO);
            ret = cuMemcpyHtoDAsync(gpuAddr(dev), tmp, size, streamToDevice);
            _pendingToDevice = true;
            unlock();
        }
    } else {
        lock();
        ret = cuMemcpyHtoD(gpuAddr(dev), host, size);
        unlock();
    }

    exitFunction();
    return error(ret);
}

gmacError_t
Context::copyToHost(void *host, const void *dev, size_t size)
{
    enterFunction(FuncAccDeviceHost);
    gmac::Context *ctx = gmac::Context::current();

    TRACE("Copy to host: %p to %p", dev, host);
    CUresult ret = CUDA_SUCCESS;
    if (_gpu.async()) {
        size_t bufferSize = ctx->bufferPageLockedSize();
        void * tmp        = ctx->bufferPageLocked();

        size_t left = size;
        off_t  off  = 0;
        while (left != 0) {
            size_t bytes = left < bufferSize? left: bufferSize;
            lock();
            pushEventState(IORead, paraver::Accelerator, 0x10000000 + _id, AcceleratorIO);
            ret = cuMemcpyDtoHAsync(tmp, gpuAddr(((char *) dev) + off), bytes, streamToHost);
            CBREAK(ret == CUDA_SUCCESS, unlock());
            ret = cuStreamSynchronize(streamToHost);
            popEventState(paraver::Accelerator, 0x10000000 + _id);
            unlock();
            CBREAK(ret == CUDA_SUCCESS);
            memcpy(((char *) host) + off, tmp, bytes);

            left -= bytes;
            off  += bytes;
        }
    } else {
        lock();
        ret = cuMemcpyDtoH(host, gpuAddr(dev), size);
        unlock();
    }

    exitFunction();
    return error(ret);
}

gmacError_t
Context::copyDevice(void *dst, const void *src, size_t size) {
    enterFunction(FuncAccDeviceDevice);
    lock();

    CUresult ret;
    ret = cuMemcpyDtoD(gpuAddr(dst), gpuAddr(src), size);

    unlock();
    exitFunction();
    return error(ret);
}

gmacError_t
Context::memset(void *addr, int i, size_t n)
{
	CUresult ret = CUDA_SUCCESS;
	unsigned char c = i & 0xff;
	lock();
	if((n % 4) == 0) {
		unsigned m = c | (c << 8);
		m |= (m << 16);
		ret = cuMemsetD32(gpuAddr(addr), m, n / 4);
	}
	else if((n % 2) == 0) {
		unsigned short s = c | (c << 8);
		ret = cuMemsetD16(gpuAddr(addr), s, n / 2);
	}
	else {
		ret = cuMemsetD8(gpuAddr(addr), c, n);
	}
	unlock();
	return error(ret);
}

gmac::KernelLaunch *
Context::launch(gmacKernel_t addr)
{
    gmac::Kernel * k = kernel(addr);
    ASSERT(k != NULL);
    _call._stream = streamLaunch;
    gmac::KernelLaunch * l = k->launch(_call);

    if (!_releasedAll) {
        if (static_cast<memory::RegionSet *>(l)->size() == 0) {
            _releasedRegions.clear();
            _releasedAll = true;
        } else {
            _releasedRegions.insert(l->begin(), l->end());
        }
    }
    //! \todo Move this to Kernel.cpp
    _pendingKernel = true;

    return l;
}

const Variable *
Context::constant(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
    for(m = _modules.begin(); m != _modules.end(); m++) {
        const Variable *var = (*m)->constant(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Variable *
Context::variable(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
    for(m = _modules.begin(); m != _modules.end(); m++) {
        const Variable *var = (*m)->variable(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Texture *
Context::texture(gmacTexture_t key) const
{
    ModuleVector::const_iterator m;
    for(m = _modules.begin(); m != _modules.end(); m++) {
        const Texture *tex = (*m)->texture(key);
        if(tex != NULL) return tex;
    }
    return NULL;
}


void
Context::flush(const char * kernel)
{
    //
#ifdef USE_VM
	ModuleVector::const_iterator m;
	for(m = _modules.begin(); pageTable == NULL && m != modules.end(); m++) {
		pageTable = m->first->pageTable();
	}
	ASSERT(pageTable != NULL);
	if(pageTable == NULL) return;

	devicePageTable.ptr = mm().pageTable().flush();
	devicePageTable.shift = mm().pageTable().getTableShift();
	devicePageTable.size = mm().pageTable().getTableSize();
	devicePageTable.page = mm().pageTable().getPageSize();
	ASSERT(devicePageTable.ptr != NULL);

	lock();
	CUresult ret = cuMemcpyHtoD(pageTable->ptr, &devicePageTable, sizeof(devicePageTable));
	ASSERT(ret == CUDA_SUCCESS);
	unlock();
#endif
}

void
Context::invalidate()
{
#ifdef USE_VM
	ModuleVector::const_iterator m;
	for(m = _modules.begin(); pageTable == NULL && m != _modules.end(); m++) {
		pageTable = m->first->pageTable();
	}
	ASSERT(pageTable != NULL);
	if(pageTable == NULL) return;

	mm().pageTable().invalidate();
#endif
}

}}
