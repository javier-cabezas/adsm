#include "Context.h"

#include <assert.h>

namespace gmac { namespace gpu {

MUTEX(Context::global);

Context::Context(const Context &root, GPU &gpu) :
	gmac::Context(gpu),
	gpu(gpu), _sp(0)
{
	setup();
	lock();
	ModuleMap::const_iterator m;
	for(m = root.modules.begin(); m != root.modules.end(); m++) {
		Module *module = new Module(*m->first);
		modules.insert(ModuleMap::value_type(module, m->second));
	}
	unlock();
	TRACE("Cloned GPU context [%p]", this);
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

}}
