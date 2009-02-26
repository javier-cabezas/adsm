#ifndef __LAZYMANAGER_H_
#define __LAZYMANAGER_H_

#include "MemManager.h"
#include "MemRegion.h"

namespace icuda {

//! Manager that Moves Memory Regions Lazily
class LazyManager : public MemManager, public MemHandler {
protected:
	HASH_MAP<void *, ProtRegion *> memMap;
public:
	virtual bool alloc(void *addr, size_t count);
	virtual void release(void *addr);
	virtual void execute(void);
	virtual void sync(void);

	virtual void read(ProtRegion *region, void *addr);
	virtual void write(ProtRegion *region, void *addr);
};

};

#endif
