#include "BatchManager.h"

#include "debug.h"

#include <cuda_runtime.h>

namespace gmac {
void BatchManager::execute(void)
{
	HASH_MAP<void *, size_t>::const_iterator i;
	for(i = memMap.begin(); i != memMap.end(); i++) {
		cudaMemcpy(i->first, i->first, i->second, cudaMemcpyHostToDevice);
	}
}

void BatchManager::sync(void)
{
	HASH_MAP<void *, size_t>::const_iterator i;
	for(i = memMap.begin(); i != memMap.end(); i++) {
		cudaMemcpy(i->first, i->first, i->second, cudaMemcpyDeviceToHost);
	}
}
};
