#include "Context.h"

#include <cassert>

namespace gmac { namespace gpu {

#ifdef USE_VM
const char *Context::pageTableSymbol = "__pageTable";
#endif

Context::Call::Call(dim3 grid, dim3 block, size_t shared, size_t tokens) :
    grid(grid),
    block(block),
    shared(shared),
    tokens(tokens)
{}

void
Context::setup()
{
    mutex = new util::Lock(paraver::ctxLocal);
    CUcontext tmp;
    assert(cuDeviceComputeCapability(&major, &minor, gpu.device()) ==
            CUDA_SUCCESS);
    unsigned int flags = 0;
    if(major > 0 && minor > 0) flags |= CU_CTX_MAP_HOST;
    CUresult ret = cuCtxCreate(&ctx, flags, gpu.device());
    if(ret != CUDA_SUCCESS)
        FATAL("Unable to create CUDA context %d", ret);
    assert(cuCtxPopCurrent(&tmp) == CUDA_SUCCESS);

    enable();
}

void
Context::setupStreams()
{
    if (gpu.async()) {
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

    if (gpu.async()) {
        TRACE("Using page locked memory: %zd\n", _bufferPageLockedSize);
        assert(cuMemHostAlloc(&_bufferPageLocked, paramBufferPageLockedSize, CU_MEMHOSTALLOC_PORTABLE) == CUDA_SUCCESS);
        _bufferPageLockedSize = paramBufferPageLockedSize;
    } else {
        _bufferPageLocked     = NULL;
        _bufferPageLockedSize = 0;
    }
}

Context::Context(Accelerator &gpu) :
    gmac::Context(gpu), gpu(gpu), _sp(0)
#ifdef USE_VM
    , pageTable(NULL)
#endif
{
	setup();

    lock();
    setupStreams();
    unlock();

    TRACE("New Accelerator context [%p]", this);
}

Context::Context(const Context &root, Accelerator &gpu) :
	gmac::Context(gpu),
	gpu(gpu), _sp(0)
#ifdef USE_VM
	,pageTable(NULL)
#endif
{
	setup();
	lock();
	ModuleMap::const_iterator m;
	for(m = root.modules.begin(); m != root.modules.end(); m++) {
		Module *module = new Module(*m->first);
		modules.insert(ModuleMap::value_type(module, m->second));
	}
	hostMem = root.hostMem;
    setupStreams();
    unlock();
	TRACE("Cloned Accelerator context [%p]", this);
}

Context::~Context()
{
    TRACE("Remove Accelerator context [%p]", this);
    delete mutex;
    if (gpu.async()) {
        cuStreamDestroy(streamLaunch);
        cuStreamDestroy(streamToDevice);
        cuStreamDestroy(streamToHost);
        cuStreamDestroy(streamDevice);
    }
    cuCtxDestroy(ctx); 
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
Context::free(void *addr)
{
    lock();
    CUresult ret = cuMemFree(gpuAddr(addr));
    unlock();
    return error(ret);
}

gmacError_t Context::hostAlloc(void **host, void **device, size_t size)
{
	zero(host);
	CUresult ret = CUDA_SUCCESS;
	lock();
    if (device != NULL) {
        zero(device);
        ret = cuMemHostAlloc(host, size, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
        if(ret == CUDA_SUCCESS) {
            assert(cuMemHostGetDevicePointer((CUdeviceptr *)device, *host, 0) == CUDA_SUCCESS);
        }
    } else {
        ret = cuMemHostAlloc(host, size, CU_MEMHOSTALLOC_PORTABLE);
    }
	hostMem.insert(AddressMap::value_type(*host, *host));
	unlock();
	return error(ret);
}


gmacError_t Context::hostMemAlign(void **host, void **device, size_t size)
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
		assert(cuMemHostGetDevicePointer((CUdeviceptr *)&dev, ptr, 0) == CUDA_SUCCESS);
	}
	*host = (void *)((uint8_t *)ptr + offset);
	hostMem.insert(AddressMap::value_type(*host, ptr));
	unlock();
	*device = (void *)(dev + offset);
	return error(ret);
}


gmacError_t Context::hostMap(void *host, void **device, size_t size)
{
	zero(device);
	void *ptr = NULL;
	CUdeviceptr dev = 0;
	size_t pageSize = mm().pageTable().getPageSize();
	assert(((unsigned long)host & (pageSize - 1)) == 0);
	proc->addShared(host, size);
	lock();
	AddressMap::const_iterator i = hostMem.find(host);
	assert(i != hostMem.end());
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
	assert(i != hostMem.end());
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
    if (gpu.async()) {
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
    if (gpu.async()) {
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

gmacError_t Context::memset(void *addr, int i, size_t n)
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


gmacError_t Context::launch(const char *kernel)
{
	assert(_calls.empty() == false);
	Call c = _calls.back();
	_calls.pop_back();
	size_t count = _sp;
	_sp = 0;

	const Function *f = function(kernel);
	assert(f != NULL);

	lock();
	// Set-up parameters
	CUresult ret = cuParamSetv(f->fun, 0, _stack, count);
	if(ret != CUDA_SUCCESS) {
		unlock();
		return error(ret);
	}
	if((ret = cuParamSetSize(f->fun, count)) != CUDA_SUCCESS) {
		unlock();
		return error(ret);
	}

#if 0
	// Set-up textures
	Textures::const_iterator t;
	for(t = _textures.begin(); t != _textures.end(); t++) {
		cuParamSetTexRef(f->fun, CU_PARAM_TR_DEFAULT, *(*t));
	}
#endif

	// Set-up shared size
	if((ret = cuFuncSetSharedSize(f->fun, c.shared)) != CUDA_SUCCESS) {
		unlock();
		return error(ret);
	}

	if((ret = cuFuncSetBlockShape(f->fun, c.block.x, c.block.y, c.block.z))
			!= CUDA_SUCCESS) {
		unlock();
		return error(ret);
	}

	ret = cuLaunchGridAsync(f->fun, c.grid.x, c.grid.y, streamLaunch);
	unlock();
	return error(ret);
}

Module *
Context::load(void *fatBin)
{
    lock();
    Module *module = new Module(fatBin);
    modules.insert(ModuleMap::value_type(module, fatBin));
    unlock();
    return module;
}

void
Context::unload(Module *mod)
{
    ModuleMap::iterator m = modules.find(mod);
    assert(m != modules.end());
    lock();
    delete m->first;
    modules.erase(m);
    unlock();
}

const Function *
Context::function(const char *name) const
{
    ModuleMap::const_iterator m;	
    for(m = modules.begin(); m != modules.end(); m++) {
        const Function *func = m->first->function(name);
        if(func != NULL) return func;
    }
    return NULL;
}

const Variable *
Context::constant(const char *name) const
{
    ModuleMap::const_iterator m;
    for(m = modules.begin(); m != modules.end(); m++) {
        const Variable *var = m->first->constant(name);
        if(var != NULL) return var;
    }
    return NULL;
}


void
Context::flush()
{
#ifdef USE_VM
	ModuleMap::const_iterator m;
	for(m = modules.begin(); pageTable == NULL && m != modules.end(); m++) {
		pageTable = m->first->pageTable();
	}
	assert(pageTable != NULL);
	if(pageTable == NULL) return;

	devicePageTable.ptr = mm().pageTable().flush();
	devicePageTable.shift = mm().pageTable().getTableShift();
	devicePageTable.size = mm().pageTable().getTableSize();
	devicePageTable.page = mm().pageTable().getPageSize();
	assert(devicePageTable.ptr != NULL);
	
	lock();
	CUresult ret = cuMemcpyHtoD(pageTable->ptr, &devicePageTable, sizeof(devicePageTable));
	assert(ret == CUDA_SUCCESS);
	unlock();
#endif
}

void Context::invalidate()
{
#ifdef USE_VM
	ModuleMap::const_iterator m;
	for(m = modules.begin(); pageTable == NULL && m != modules.end(); m++) {
		pageTable = m->first->pageTable();
	}
	assert(pageTable != NULL);
	if(pageTable == NULL) return;

	mm().pageTable().invalidate();
#endif
}


}}
