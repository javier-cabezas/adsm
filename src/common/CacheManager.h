#ifndef __CACHEMANAGER_H_
#define __CACHEMANAGER_H_

#include "MemManager.h"
#include "MemRegion.h"

#include <vector>

namespace icuda {

class CacheRegion : public MemRegion {
protected:
	MemHandler &memHandler;
	std::vector<ProtRegion *> cache;

	size_t cacheLine;
public:
	CacheRegion(MemHandler &, void *, size_t, size_t);
	~CacheRegion();

	std::vector<ProtRegion *> &getCache() { return cache; }
};

class CacheManager : public MemManager, public MemHandler {
protected:
	static const size_t lineSize = 64;
	static const size_t lruSize = 2;
	size_t pageSize;

	HASH_MAP<void *, CacheRegion *> memMap;
	std::list<ProtRegion *> lru;
	ProtRegion *writeBuffer;

	void writeBack();
	void dmaToDevice(std::vector<ProtRegion *> &);
	
public:
	CacheManager();
	virtual bool alloc(void *addr, size_t size);
	virtual void release(void *addr);
	virtual void execute(void);
	virtual void sync(void);

	virtual void read(ProtRegion *region, void *addr);
	virtual void write(ProtRegion *region, void *addr);
};

};

#endif
