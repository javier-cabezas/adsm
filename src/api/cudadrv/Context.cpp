#include "Context.h"

#include <assert.h>

namespace gmac { namespace gpu {

#ifdef USE_VM
const char *Context::pageTableSymbol = "__pageTable";
#endif

Context::Context(const Context &root, GPU &gpu) :
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
	unlock();
	TRACE("Cloned GPU context [%p]", this);
}


gmacError_t Context::hostAlloc(void **host, void **device, size_t size)
{
	zero(host); zero(device);
	CUresult ret = CUDA_SUCCESS;
	lock();
	ret = cuMemHostAlloc(host, size, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
	if(ret == CUDA_SUCCESS) {
		assert(cuMemHostGetDevicePointer((CUdeviceptr *)device, *host, 0) == CUDA_SUCCESS);
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

gmacError_t Context::hostFree(void *addr)
{
	lock();
	AddressMap::iterator i = hostMem.find(addr);
	assert(i != hostMem.end());
	CUresult ret = cuMemFreeHost(i->second);
	hostMem.erase(i);
	unlock();
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
	size_t count = _sp - c.stack;
	_sp = c.stack;

	const Function *f = function(kernel);
	assert(f != NULL);

	lock();
	// Set-up parameters
	CUresult ret = cuParamSetv(f->fun, 0, &_stack[c.stack], count);
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

	ret = cuLaunchGrid(f->fun, c.grid.x, c.grid.y);
	unlock();
	return error(ret);
}

void Context::flush()
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
