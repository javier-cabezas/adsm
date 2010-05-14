#ifndef __MANAGER_IPP_
#define __MANAGER_IPP_

namespace gmac { namespace memory {

inline void
Manager::insert(Region *r)
{
    current()->insert(r);
}

inline Context *
Manager::owner(const void * addr)
{
	//Region *region = current()->find<Region>(addr);
	Region *region = Map::globalFind(addr);
	if (region == NULL) region = Map::sharedFind(addr);
	if(region == NULL) return NULL;
	return region->owner();
}

inline Map *
Manager::current()
{
    return &Context::current()->mm();
}

inline void
Manager::insertVirtual(void *cpuPtr, void *devPtr, size_t count)
{
	insertVirtual(Context::current(), cpuPtr, devPtr, count);
}

inline void
Manager::removeVirtual(void *cpuPtr, size_t count)
{
	removeVirtual(Context::current(), cpuPtr, count);
}

inline const PageTable &
Manager::pageTable() const
{
    return Context::current()->mm().pageTable();
}

inline const void *
Manager::ptr(Context *ctx, const void *addr)
{
    PageTable &pageTable = ctx->mm().pageTable();
    const void *ret = (const void *)pageTable.translate(addr);
    if(ret == NULL) ret = proc->translate(addr);
    return ret;
}

inline const void *
Manager::ptr(const void *addr)
{
    return ptr(::manager->owner(addr), addr);
}

inline void *
Manager::ptr(Context *ctx, void *addr)
{
    PageTable &pageTable = ctx->mm().pageTable();
    void *ret = (void *)pageTable.translate(addr);
    if(ret == NULL) ret = proc->translate(addr);
    return ret;
}

inline void *
Manager::ptr(void *addr)
{
    return ptr(::manager->owner(addr), addr);
}

inline int
Manager::defaultProt()
{
    return PROT_READ | PROT_WRITE;
}

}};
#endif
