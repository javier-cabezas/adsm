
#ifndef __MEMORY_POSIX_MEMORY_IPP_
#define __MEMORY_POSIX_MEMORY_IPP_

inline int
Memory::protect(void *addr, size_t count, int prot)
{
    if (prot == PROT_NONE) {
        util::Logger::TRACE("Invalidating %p:%p", addr, (uint8_t *) addr + count - 1);
    }
    util::Logger::TRACE("Setting memory permisions to %d @ %p - %p", prot, addr, (uint8_t *)addr + count);
    return mprotect(addr, count, prot);
}

#endif
