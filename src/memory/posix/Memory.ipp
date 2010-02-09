
#ifndef __MEMORY_POSIX_MEMORY_IPP_
#define __MEMORY_POSIX_MEMORY_IPP_

inline int
Memory::protect(void *addr, size_t count, int prot)
{
    return mprotect(addr, count, prot);
}

#endif
