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
Manager::malloc(Context * ctx, void ** addr, size_t count)
{
    gmacError_t ret;
    void *devAddr;
    void *cpuAddr;
    *addr = NULL;

	PageTable &pageTable = ctx->mm().pageTable();

    // Allocate device memory
    ret = ctx->malloc(&devAddr, count, pageTable.getPageSize());
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
    TRACE("Alloc %p (%zd bytes)", cpuAddr, count);
    *addr = cpuAddr;
    return gmacSuccess;
}

gmacError_t
Manager::halloc(Context * ctx, void ** addr, size_t count)
{
    gmacError_t ret;
    *addr = NULL;
	PageTable &pageTable = ctx->mm().pageTable();

    // Allocate page-locked memory. We currently rely on the backend
    // to allocate this memory
    ret = ctx->mallocPageLocked(addr, count, pageTable.getPageSize());
    return ret;
}


#ifdef USE_GLOBAL_HOST
gmacError_t
Manager::globalMalloc(Context * ctx, void ** addr, size_t count)
{
    gmacError_t ret;
    void *devAddr;
    void *cpuAddr;

    *addr = NULL;
	PageTable &pageTable = ctx->mm().pageTable();

    // Allocate page-locked memory. We currently rely on the backend
    // to allocate this memory
    ret = ctx->mallocPageLocked(&cpuAddr, count, pageTable.getPageSize());

    // Successful device memory allocation?
	if(ret != gmacSuccess) {
		return ret;
	}
    // Create a new (shared) region
    Region * r = newRegion(cpuAddr, count, true);
    r->lockWrite();
    // Map the page-locked memory in all the contexts
    ContextList::const_iterator i;
    ContextList & contexts = proc->contexts();
    contexts.lockRead();
    for(i = contexts.begin(); i != contexts.end(); i++) {
        Context * _ctx = *i;
        //! \todo Check the return value of the function
        ret = _ctx->mapToDevice(cpuAddr, &devAddr, count);
        map(_ctx, r, devAddr);
    }
    contexts.unlock();
    // Insert the region in the global shared map
    Map::addShared(r);
    r->unlock();
    *addr = cpuAddr;
    TRACE("Alloc %p (%zd bytes)", cpuAddr, count);
    return gmacSuccess;
}
#else
gmacError_t
Manager::globalMalloc(Context * ctx, void ** addr, size_t count)
{
    gmacError_t ret;
    void *devAddr;
    void *cpuAddr;

    // Allocate (or map) host memory
    cpuAddr = mapToHost(&devAddr, count, defaultProt());
    if (cpuAddr == NULL) // Failed!
        return gmacErrorMemoryAllocation;

    // Create a new (shared) region
    Region * r = newRegion(cpuAddr, count, true);
    r->lockWrite();
    ContextList::const_iterator i;
    ContextList & contexts = proc->contexts();
    contexts.lockRead();
    for(i = contexts.begin(); i != contexts.end(); i++) {
        Context * ctx = *i;
        PageTable &pageTable = ctx->mm().pageTable();
        // Allocate device memory. We currently rely on the backend
        // to allocate this memory
        ret = ctx->malloc(&devAddr, count, pageTable.getPageSize());
        if(ret != gmacSuccess) goto cleanup;
        map(ctx, r, devAddr);
    }
    Map::addShared(r);
    r->unlock();
    *addr = cpuAddr;
    return gmacSuccess;
cleanup:
    Context *last = *i;
    for(i = proc->contexts().begin(); *i != last; i++) {
        Context * ctx = *i;
        ctx->free(ptr(ctx, cpuAddr));
        unmap(ctx, r);
    }

    hostUnmap(r->start(), r->size());
    r->unlock();
    delete r;

    return ret;
}
#endif

gmacError_t
Manager::free(Context * ctx, void * addr)
{
    gmacError_t ret;
    Region * r = ctx->mm().find<Region>(addr);
    if (r == NULL) {
        return gmacErrorInvalidValue;
    }
    r->lockWrite();
	// If it is a shared global structure and nobody is accessing
	// it anymore, release the host memory
	if(r->shared()) {
#ifdef USE_GLOBAL_HOST 
        ctx->hostFree(addr);
#else
        hostUnmap(r->start(), r->size());
#endif
        ContextList::const_iterator i;
        ContextList & contexts = proc->contexts();
        contexts.lockRead();
        for(i = contexts.begin(); i != contexts.end(); i++) {
            Context * ctx = *i;
            // Free memory in the device
#ifndef USE_GLOBAL_HOST 
            ret = ctx->free(ptr(ctx, addr));
            ASSERT(ret == gmacSuccess);
#endif
            unmap(ctx, r);
        }
        Map::removeShared(r);
	}
	else {
		ret = ctx->free(ptr(ctx, addr));
        ASSERT(ret == gmacSuccess);
        unmap(ctx, r);
	}
    r->unlock();
    delete r;

    return gmacSuccess;
}

gmacError_t
Manager::hfree(Context * ctx, void * addr)
{
    gmacError_t ret;
    ret = ctx->hostFree(addr);
    return ret;
}

Region *Manager::remove(void *addr)
{
    Context * ctx = Context::current();
    Map & map = ctx->mm();
    map.lockWrite();
	Region *ret = map.remove(addr);
    map.unlock();
	return ret;
}

void Manager::insertVirtual(Context *ctx, void *cpuPtr, void *devPtr, size_t count)
{
#ifndef USE_MMAP
#ifndef USE_OPENCL
	TRACE("Virtual Request %p -> %p", cpuPtr, devPtr);
	PageTable &pageTable = ctx->mm().pageTable();
	ASSERT(((unsigned long)cpuPtr & (pageTable.getPageSize() -1)) == 0);
	uint8_t *devAddr = (uint8_t *)devPtr;
	uint8_t *cpuAddr = (uint8_t *)cpuPtr;
	TRACE("Page Table Request %p -> %p", cpuAddr, devAddr);
	for(size_t off = 0; off < count; off += pageTable.getPageSize())
		pageTable.insert(cpuAddr + off, devAddr + off);
#endif
#endif
}

void Manager::removeVirtual(Context *ctx, void *cpuPtr, size_t count)
{
#ifndef USE_MMAP
#ifndef USE_OPENCL
	uint8_t *cpuAddr = (uint8_t *)cpuPtr;
	PageTable &pageTable = ctx->mm().pageTable();
	count += ((unsigned long)cpuPtr & (pageTable.getPageSize() -1));
	for(size_t off = 0; off < count; off += pageTable.getPageSize())
		pageTable.remove(cpuAddr + off);
#endif
#endif
}

void Manager::unmap(Context *ctx, Region *r)
{
	ASSERT(r != NULL);
	r->unrelate(ctx);
	removeVirtual(ctx, r->start(), r->size());
    if (r->shared() == false) {
        Map & map = ctx->mm();
        map.remove(r->start());
    }
}

void
Manager::initShared(Context * ctx)
{
#ifndef USE_MMAP
	PageTable &pageTable = ctx->mm().pageTable();
    RegionMap::iterator i;
    RegionMap &shared = Map::shared();
    shared.lockRead();
    for(i = shared.begin(); i != shared.end(); i++) {
        Region * r = i->second;
        r->lockWrite();
        TRACE("Mapping Shared Region %p (%zd bytes)", r->start(), r->size());
        void *devPtr;
#ifdef USE_GLOBAL_HOST
        TRACE("Using Host Translation");
        gmacError_t ret = ctx->mapToDevice(r->start(), &devPtr, r->size());
#else
        gmacError_t ret = ctx->malloc(&devPtr, r->size(), pageTable.getPageSize());
#endif
        ASSERT(ret == gmacSuccess);
        map(ctx, r, devPtr);
        r->unlock();
    }
    shared.unlock();
#endif
}

void
Manager::syncToHost()
{
    Map::const_iterator i;
    Map * m = current();
    m->lockRead();
    for(i = m->begin(); i != m->end(); i++) {
        Region * r = i->second;
        r->lockWrite();
        r->syncToHost();
        r->unlock();
    }
    m->unlock();
}

gmacError_t
Manager::freeDevice(std::vector<void *> & oldAddr)
{
    gmacError_t ret;
    Context * ctx = Context::current();
    std::vector<void *>::const_iterator i;
    for(i = oldAddr.begin(); i != oldAddr.end(); i++) {
        ret = ctx->free(*i);
        ASSERT(ret == gmacSuccess);
    }

    return gmacSuccess;
}

std::vector<void *>
Manager::reallocDevice()
{
    std::vector<void *> oldAddr;

    void *devAddr;
    gmacError_t ret;
    Context * ctx = Context::current();

	PageTable &pageTable = ctx->mm().pageTable();

    Map::const_iterator i;
    Map * m = current();
    m->lockRead();
    for(i = m->begin(); i != m->end(); i++) {
        Region * r = i->second;
        r->lockWrite();
        // If it is a shared global structure and nobody is accessing
        // it anymore, release the host memory
        if(r->shared()) {
            // Free memory in the device
#ifdef USE_GLOBAL_HOST
            // Map memory in the device
            ret = ctx->mapToDevice(r->start(), &devAddr, r->size());
            ASSERT(ret == gmacSuccess);
#else
            oldAddr.push_back(ptr(ctx, r->start()));
            // Allocate device memory
            ret = ctx->malloc(&devAddr, r->size(), pageTable.getPageSize());
            ASSERT(ret == gmacSuccess);
#endif
            unmap(ctx, r);
            map(ctx, r, devAddr);
        }
        else {
            oldAddr.push_back(ptr(ctx, r->start()));
            // Allocate device memory
            ret = ctx->malloc(&devAddr, r->size(), pageTable.getPageSize());
            ASSERT(ret == gmacSuccess);
            unmap(ctx, r);
            // Insert mapping in the page table
            insertVirtual(r->start(), devAddr, r->size());
            insert(r);
        }
        r->unlock();
    }
    m->unlock();

    return oldAddr;
}

gmacError_t
Manager::touchAll()
{
    gmacError_t ret;
    Context * ctx = Context::current();

    Map::const_iterator i;
    Map * m = current();
    m->lockRead();
    for(i = m->begin(); i != m->end(); i++) {
        Region * r = i->second;
        r->lockWrite();
        bool correct = touch(r);
        ASSERT(correct);
        r->unlock();
    }
    m->unlock();

    return gmacSuccess;
}

}}
