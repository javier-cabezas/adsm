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
    mutex = new util::Lock(paraver::ctxLocal);
    _ctx = _gpu.createCUDAContext();
    enable();
}

void
Context::setupStreams()
{
    if (_gpu.async()) {
        CUresult ret;
        ret = cuStreamCreate(&streamLaunch, 0);
        if(ret != CUDA_SUCCESS)
            FATAL("Unable to create CUDA stream %d", ret);
        ret = cuStreamCreate(&streamToDevice, 0);
        if(ret != CUDA_SUCCESS)
            FATAL("Unable to create CUDA stream %d", ret);
        ret = cuStreamCreate(&streamToHost, 0);
        if(ret != CUDA_SUCCESS)
            FATAL("Unable to create CUDA stream %d", ret);
        ret = cuStreamCreate(&streamDevice, 0);
        if(ret != CUDA_SUCCESS)
            FATAL("Unable to create CUDA stream %d", ret);
    }

    if (_gpu.async()) {
        CUresult ret = cuMemHostAlloc(&_bufferPageLocked, paramBufferPageLockedSize, CU_MEMHOSTALLOC_PORTABLE);
        ASSERT(ret == CUDA_SUCCESS);
        _bufferPageLockedSize = paramBufferPageLockedSize;
        TRACE("Using page locked memory: %zd\n", _bufferPageLockedSize);
    } else {
        _bufferPageLocked     = NULL;
        _bufferPageLockedSize = 0;
    }
}

Context::Context(Accelerator &gpu) :
    gmac::Context(gpu),
    _gpu(gpu),
    _call(dim3(0), dim3(0), 0, 0)
#ifdef USE_VM
    , pageTable(NULL)
#endif
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
    delete mutex;
}

gmacError_t
Context::malloc(void **addr, size_t size) {
    zero(addr);
    lock();
    size += mm().pageTable().getPageSize();
    CUdeviceptr ptr = 0;
    CUresult ret = cuMemAlloc(&ptr, size);
    if(ptr % mm().pageTable().getPageSize()) {
        ptr += mm().pageTable().getPageSize() -
            (ptr % mm().pageTable().getPageSize());
    }
    *addr = (void *)ptr;
    unlock();
    return error(ret);
}

gmacError_t
Context::halloc(void **addr, size_t size) {
    zero(addr);
    lock();
    size += mm().pageTable().getPageSize();
    uint8_t * ptr = 0;
    CUresult ret = cuMemHostAlloc((void **) &ptr, size, CU_MEMHOSTALLOC_PORTABLE);
    if(uint64_t(ptr) % mm().pageTable().getPageSize()) {
        ptr += mm().pageTable().getPageSize() -
            (uint64_t(ptr) % mm().pageTable().getPageSize());
    }
    *addr = (void *)ptr;
    unlock();
    return error(ret);
}


gmacError_t
Context::free(void *addr)
{
    lock();
    CUresult ret = cuMemFree(gpuAddr(addr));
    unlock();
    return error(ret);
}

gmacError_t
Context::hostAlloc(void **host, void **device, size_t size)
{
	zero(host);
	CUresult ret = CUDA_SUCCESS;
	lock();
    if (device != NULL) {
        zero(device);
        ret = cuMemHostAlloc(host, size, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
        if(ret == CUDA_SUCCESS) {
            ret = cuMemHostGetDevicePointer((CUdeviceptr *)device, *host, 0);
            ASSERT(ret == CUDA_SUCCESS);
        }
    } else {
        ret = cuMemHostAlloc(host, size, CU_MEMHOSTALLOC_PORTABLE);
    }
	hostMem.insert(AddressMap::value_type(*host, *host));
	unlock();
	return error(ret);
}


gmacError_t
Context::hostMemAlign(void **host, void **device, size_t size)
{
	zero(host); zero(device);
	void *ptr = NULL;
	CUdeviceptr dev = 0;
	size_t pageSize = mm().pageTable().getPageSize();
	size_t offset = 0;
	CUresult ret = CUDA_SUCCESS;
	lock();
	ret = cuMemHostAlloc(&ptr, size, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
	if((unsigned long)ptr & (pageSize - 1)) {
		size += pageSize;
		cuMemFreeHost(ptr);
		ret = cuMemHostAlloc(&ptr, size, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
		offset = pageSize - ((unsigned long)ptr & (pageSize - 1));
		*host = (void *)((uint8_t *)ptr + offset);
	}
	if(ret == CUDA_SUCCESS) {
		ret = cuMemHostGetDevicePointer((CUdeviceptr *)&dev, ptr, 0);
        ASSERT(ret == CUDA_SUCCESS);
	}
	*host = (void *)((uint8_t *)ptr + offset);
	hostMem.insert(AddressMap::value_type(*host, ptr));
	unlock();
	*device = (void *)(dev + offset);
	return error(ret);
}


gmacError_t
Context::hostMap(void *host, void **device, size_t size)
{
	zero(device);
	void *ptr = NULL;
	CUdeviceptr dev = 0;
	size_t pageSize = mm().pageTable().getPageSize();
	ASSERT(((unsigned long)host & (pageSize - 1)) == 0);
	lock();
	AddressMap::const_iterator i = hostMem.find(host);
	ASSERT(i != hostMem.end());
	CUresult ret = cuMemHostGetDevicePointer((CUdeviceptr *)&dev, i->second, 0);
	size_t offset = (uint8_t *)i->first - (uint8_t *)i->second;
	unlock();
	*device = (void *)(dev + offset);
	return error(ret);
}

gmacError_t
Context::hostFree(void *addr)
{
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
    enterFunction(accHostDeviceCopy);
    gmac::Context *ctx = gmac::Context::current();

    CUresult ret;
    if (_gpu.async()) {
        size_t bufferSize = ctx->bufferPageLockedSize();
        void * tmp        = ctx->bufferPageLocked();

        size_t left = size;
        off_t  off  = 0;
        while (left != 0) {
            size_t bytes = left < bufferSize? left: bufferSize;
            memcpy(tmp, ((char *) host) + off, bytes);
            lock();
            ret = cuMemcpyHtoDAsync(gpuAddr(((char *) dev) + off), tmp, bytes, streamToDevice);
            if (ret != CUDA_SUCCESS) { unlock(); goto done; }
            ret = cuStreamSynchronize(streamToDevice);
            unlock();
            if (ret != CUDA_SUCCESS) goto done;

            left -= bytes;
            off  += bytes;
        }
    } else {
        lock();
        ret = cuMemcpyHtoD(gpuAddr(dev), host, size);
        unlock();
    }

done:
    exitFunction();
    return error(ret);
}

gmacError_t
Context::copyToHost(void *host, const void *dev, size_t size)
{
    enterFunction(accDeviceHostCopy);
    gmac::Context *ctx = gmac::Context::current();

    CUresult ret;
    if (_gpu.async()) {
        size_t bufferSize = ctx->bufferPageLockedSize();
        void * tmp        = ctx->bufferPageLocked();

        size_t left = size;
        off_t  off  = 0;
        while (left != 0) {
            size_t bytes = left < bufferSize? left: bufferSize;
            lock();
            ret = cuMemcpyDtoHAsync(tmp, gpuAddr(((char *) dev) + off), bytes, streamToHost);
            if (ret != CUDA_SUCCESS) { unlock(); goto done; }
            ret = cuStreamSynchronize(streamToHost);
            unlock();
            if (ret != CUDA_SUCCESS) goto done;
            memcpy(((char *) host) + off, tmp, bytes);

            left -= bytes;
            off  += bytes;
        }
    } else {
        lock();
        ret = cuMemcpyDtoH(host, gpuAddr(dev), size);
        unlock();
    }

done:
    exitFunction();
    return error(ret);
}

gmacError_t
Context::copyDevice(void *dst, const void *src, size_t size) {
    enterFunction(accDeviceDeviceCopy);
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
    _status = RUNNING;

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
