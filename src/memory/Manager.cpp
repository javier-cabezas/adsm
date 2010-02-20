#include "Manager.h"
#include "manager/BatchManager.h"
#include "manager/LazyManager.h"
#include "manager/RollingManager.h"

#include <debug.h>

#include <strings.h>


namespace gmac { namespace memory {

Region *    
Manager::newRegion(void * addr, size_t count, bool shared)
{
    return new Region(addr, count, shared);
}

Manager *getManager(const char *managerName)
{
	if(managerName == NULL) return new manager::RollingManager();
	TRACE("Using %s Manager", managerName);
	if(strcasecmp(managerName, "None") == 0)
		return NULL;
	else if(strcasecmp(managerName, "Lazy") == 0)
		return new manager::LazyManager();
	else if(strcasecmp(managerName, "Batch") == 0)
		return new manager::BatchManager();
	return new manager::RollingManager();
}

Manager::Manager()
{
    TRACE("Memory manager starts");
}

Manager::~Manager()
{
    TRACE("Memory manager finishes");
}


gmacError_t
Manager::malloc(void ** addr, size_t count)
{
    gmacError_t ret;
    void *devAddr;
    void *cpuAddr;

    *addr = NULL;

    // Allocate device memory
    Context * ctx = Context::current();
    ret = ctx->malloc(&devAddr, count);
    // Successful device memory allocation?
	if(ret != gmacSuccess) {
		return ret;
	}
    // Allocate (or map) host memory
    cpuAddr = mapToHost(devAddr, count, defaultProt());
    if (cpuAddr == NULL) // Failed!
        return gmacErrorMemoryAllocation;
    // Insert mapping in the page table
    insertVirtual(cpuAddr, devAddr, count);
    // Create a new region
    Region * r = newRegion(cpuAddr, count, false);
    // Insert the region in the local and global memory maps
    insert(r);
    TRACE("Alloc %p (%d bytes)", cpuAddr, count);
    *addr = cpuAddr;
    return gmacSuccess;
}

#ifdef USE_GLOBAL_HOST
gmacError_t
Manager::globalMalloc(void ** addr, size_t count)
{
    gmacError_t ret;
    void *devAddr;
    void *cpuAddr;

    *addr = NULL;

    // Allocate page-locked memory. We currently rely on the backend
    // to allocate this memory
    Context * ctx = Context::current();
    ret = ctx->mallocPageLocked(&cpuAddr, count);
    // Successful device memory allocation?
	if(ret != gmacSuccess) {
		return ret;
	}
    // Create a new (shared) region
    Region * r = newRegion(cpuAddr, count, true);
    // Map the page-locked memory in all the contexts
    ContextList::const_iterator i;
    ContextList & contexts = proc->contexts();
    contexts.lockRead();
    for(i = contexts.begin(); i != contexts.end(); i++) {
        Context * ctx = *i;
        //! \todo Check the return value of the function
        ctx->mapToDevice(cpuAddr, &devAddr, count);
        remap(ctx, r, devAddr);
    }
    contexts.unlock();
    // Insert the region in the global shared map
    Map::addShared(r);
    *addr = cpuAddr;
    TRACE("Alloc %p (%d bytes)", cpuAddr, count);
    return gmacSuccess;
}
#else
gmacError_t
Manager::globalMalloc(void ** addr, size_t count)
{
#ifndef USE_MMAP
    gmacError_t ret;
    void *devAddr;
    void *cpuAddr;

    // Allocate (or map) host memory
    cpuAddr = mapToHost(devAddr, count, defaultProt());
    if (cpuAddr == NULL) // Failed!
        return gmacErrorMemoryAllocation;

    // Create a new (shared) region
    Region * r = newRegion(cpuAddr, count, true);

    ContextList::const_iterator i;
    ContextList & contexts = proc->contexts();
    contexts.lockRead();
    Process::ContextList::const_iterator i;
    for(i = proc->contexts().begin(); i != proc->contexts().end(); i++) {
        Context * ctx = Context::current();
        // Allocate device memory. We currently rely on the backend
        // to allocate this memory
        ret = ctx->malloc(&devAddr, count);
        if(ret != gmacSuccess) goto cleanup;
        manager->remap(ctx, r, devAddr);
    }
    Map::addShared(r);
    *addr = cpuAddr;
    return gmacSuccess;
cleanup:
    Context *last = *i;
    for(i = proc->contexts().begin(); *i != last; i++) {
        Context * ctx = *i;
        ctx->free(manager->ptr(ctx, cpuAddr));
        manager->unmap(ctx, *cpuAddr);
    }

    delete r;

    gmacFree(devAddr);
    exitFunction();
    __exitGmac();
    return ret;
#else
    return gmacErrorFeatureNotSupported;
#endif
}
#endif

Region *Manager::remove(void *addr)
{
    Context * ctx = Context::current();
	Region *ret = ctx->mm().remove(addr);
	return ret;
}

void Manager::insertVirtual(Context *ctx, void *cpuPtr, void *devPtr, size_t count)
{
#ifndef USE_MMAP
	TRACE("Virtual Request %p -> %p", cpuPtr, devPtr);
	PageTable &pageTable = ctx->mm().pageTable();
	ASSERT(((unsigned long)cpuPtr & (pageTable.getPageSize() -1)) == 0);
	uint8_t *devAddr = (uint8_t *)devPtr;
	uint8_t *cpuAddr = (uint8_t *)cpuPtr;
	TRACE("Page Table Request %p -> %p", cpuAddr, devAddr);
	for(size_t off = 0; off < count; off += pageTable.getPageSize())
		pageTable.insert(cpuAddr + off, devAddr + off);
#endif
}

void Manager::removeVirtual(Context *ctx, void *cpuPtr, size_t count)
{
#ifndef USE_MMAP
	uint8_t *cpuAddr = (uint8_t *)cpuPtr;
	PageTable &pageTable = ctx->mm().pageTable();
	count += ((unsigned long)cpuPtr & (pageTable.getPageSize() -1));
	for(size_t off = 0; off < count; off += pageTable.getPageSize())
		pageTable.remove(cpuAddr + off);
#endif
}

#if 0
void Manager::unmap(Context *ctx, void *cpuPtr)
{
	Region *region = Context::current()->mm().find<Region>(cpuPtr);
	ASSERT(region != NULL);
	region->unrelate(ctx);
	removeVirtual(ctx, cpuPtr, region->size());
}
#endif

void
Manager::initShared(Context * ctx)
{
    RegionMap::iterator i;
    RegionMap &shared = Map::shared();
    shared.lockRead();
    for(i = shared.begin(); i != shared.end(); i++) {
        Region * r = i->second;
        TRACE("Mapping Shared Region %p (%d bytes)", r->start(), r->size());
        void *devPtr;
#ifdef USE_GLOBAL_HOST
        TRACE("Using Host Translation");
        gmacError_t ret = ctx->mapToDevice(r->start(), &devPtr, r->size());
#else
        gmacError_t ret = ctx->malloc(&devPtr, r->size());
#endif
        ASSERT(ret == gmacSuccess);
        remap(ctx, r, devPtr);
    }
    shared.unlock();
}

}}
