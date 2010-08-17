
#ifndef __MEMORY_POSIX_MEMORY_IPP_
#define __MEMORY_POSIX_MEMORY_IPP_

inline int
Memory::protect(void *addr, size_t count, int prot)
{
    util::Logger::TRACE("Protecting %p:%p: %zd, %d", addr, (uint8_t *) addr + count - 1, count, prot);
    return mprotect(addr, count, prot);
}

#endif
