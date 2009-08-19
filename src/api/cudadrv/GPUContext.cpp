#include "GPUContext.h"

#include <assert.h>

namespace gmac {

gmacError_t GPUContext::launch(const char *kernel)
{
	assert(_calls.empty() == false);
	Call c = _calls.back();
	_calls.pop_back();
	size_t count = _sp - c.stack;

	const CUfunction *f = function(kernel);
	assert(f != NULL);

	lock();
	// Set-up parameters
	CUresult ret = cuParamSetv(*f, 0, &_stack[c.stack], count);
	if(ret != CUDA_SUCCESS) {
		release();
		return error(gpu.error(ret));
	}
	if((ret = cuParamSetSize(*f, count)) != CUDA_SUCCESS) {
		release();
		return error(gpu.error(ret));
	}

	// Set-up textures
	Textures::const_iterator t;
	for(t = _textures.begin(); t != _textures.end(); t++) {
		cuParamSetTexRef(*f, CU_PARAM_TR_DEFAULT, *(*t));
	}

	// Set-up shared size
	if((ret = cuFuncSetSharedSize(*f, c.shared)) != CUDA_SUCCESS) {
		release();
		return error(gpu.error(ret));
	}

	gmacError_t err = gpu.launch(c.grid, c.block, *f);
	release();

	return error(err);
}

};
