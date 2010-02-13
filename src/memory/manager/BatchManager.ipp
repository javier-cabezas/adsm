#ifndef __MEMORY_BATCHMANAGER_IPP_
#define __MEMORY_BATCHMANAGER_IPP_

inline void *
BatchManager::alloc(void *addr, size_t count, int attr)
{
    void *cpuAddr;

    if (attr == GMAC_MALLOC_PINNED) {
        Context * ctx = Context::current();
        void *hAddr;
        if (ctx->halloc(&hAddr, count) != gmacSuccess) return NULL;
        cpuAddr = hostRemap(addr, hAddr, count);
    } else {
        cpuAddr = hostMap(addr, count);
    }
    if (cpuAddr == NULL) return NULL;
    insertVirtual(cpuAddr, addr, count);
    insert(new Region(cpuAddr, count));
    TRACE("Alloc %p (%d bytes)", cpuAddr, count);
    return cpuAddr;
}

#endif
