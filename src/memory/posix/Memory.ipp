
#ifndef __MEMORY_POSIX_MEMORY_IPP_
#define __MEMORY_POSIX_MEMORY_IPP_

inline int
Memory::protect(void *addr, size_t count, int prot)
{
    if (prot == PROT_NONE) {
        logger.trace("Invalidating %p:%p", addr, (uint8_t *) addr + count - 1);
    }
    return mprotect(addr, count, prot);
}

#endif
