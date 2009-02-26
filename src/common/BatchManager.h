#ifndef __BATCHMANAGER_H_
#define __BATCHMANAGER_H_

#include "MemManager.h"

#include <stdint.h>

namespace icuda {
//! Batch Memory Manager

//! The Batch Memory Manager moves all data just before and
//! after a kernel call
class BatchManager : public MemManager {
protected:
	HASH_MAP<void *, size_t> memMap;
public:
	inline bool alloc(void *addr, size_t count) {
		if(map(addr, count) == MAP_FAILED) return false;
		memMap[addr] = count;
		return true;
	}
	inline void release(void *addr) {
		if(memMap.find(addr) == memMap.end()) return;
		unmap(addr, memMap[addr]);
		memMap.erase(addr);
	}
	void execute(void);
	void sync(void);
};
};
#endif
