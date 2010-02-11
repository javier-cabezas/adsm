#ifndef __MANAGER_IPP_
#define __MANAGER_IPP_

inline void
Manager::insert(Region *r)
{
    gmac::Context::current()->mm().insert(r);
}

inline Context *
Manager::owner(const void * addr)
{
	Region *region = current()->find<Region>(addr);
	if(region == NULL) return NULL;
	return region->owner();
}

inline memory::Map *
Manager::current()
{
    return &gmac::Context::current()->mm();
}

inline void
Manager::insertVirtual(void *cpuPtr, void *devPtr, size_t count)
{
	insertVirtual(gmac::Context::current(), cpuPtr, devPtr, count);
}

inline void
Manager::removeVirtual(void *cpuPtr, size_t count)
{
	removeVirtual(gmac::Context::current(), cpuPtr, count);
}

inline const memory::PageTable &
Manager::pageTable() const
{
    return gmac::Context::current()->mm().pageTable();
}

inline const void *
Manager::ptr(Context *ctx, const void *addr)
{
    memory::PageTable &pageTable = ctx->mm().pageTable();
    const void *ret = (const void *)pageTable.translate(addr);
    if(ret == NULL) ret = proc->translate(addr);
    return ret;
}

inline const void *
Manager::ptr(const void *addr)
{
    return ptr(gmac::Context::current(), addr);
}

inline void *
Manager::ptr(Context *ctx, void *addr)
{
    memory::PageTable &pageTable = ctx->mm().pageTable();
    void *ret = (void *)pageTable.translate(addr);
    if(ret == NULL) ret = proc->translate(addr);
    return ret;
}

inline void *
Manager::ptr(void *addr)
{
    return ptr(gmac::Context::current(), addr);
}

#endif
