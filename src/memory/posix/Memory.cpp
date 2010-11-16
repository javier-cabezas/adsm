#include "memory/Memory.h"
#include "memory/posix/FileMap.h"
#include "core/Mode.h"

#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

namespace gmac { namespace memory {

int ProtBits[] = {
    PROT_NONE,
    PROT_READ,
    PROT_WRITE,
    PROT_READ | PROT_WRITE
};

static FileMap Files;

int Memory::protect(void *addr, size_t count, GmacProtection prot)
{
    TRACE(GLOBAL, "Setting memory permisions to %d @ %p - %p", prot, addr, (uint8_t *)addr + count);
    int ret = mprotect(addr, count, ProtBits[prot]);
    ASSERTION(ret == 0);
    return 0;
}

void *Memory::map(void *addr, size_t count, GmacProtection prot)
{
    void *cpuAddr = NULL;
    char tmp[FILENAME_MAX];

    // Create new shared memory file
    snprintf(tmp, FILENAME_MAX, "/tmp/gmacXXXXXX");
    int fd = mkstemp(tmp);
    if(fd < 0) return NULL;
    unlink(tmp);

    if(ftruncate(fd, count) < 0) {
        close(fd);
        return NULL;
    }

    if (addr == NULL) {
        cpuAddr = mmap(addr, count, ProtBits[prot], MAP_SHARED, fd, 0);
        TRACE(GLOBAL, "Getting map: %d @ %p - %p", prot, cpuAddr, (uint8_t *)addr + count);
    } else {
        cpuAddr = addr;
        if(mmap(cpuAddr, count, ProtBits[prot], MAP_SHARED | MAP_FIXED, fd, 0) != cpuAddr) {
            close(fd);
            return NULL;
        }
        TRACE(GLOBAL, "Getting fixed map: %d @ %p - %p", prot, addr, (uint8_t *)addr + count);
    }

    if(Files.insert(fd, cpuAddr, count) == false) {
        munmap(cpuAddr, count);
        close(fd);
        return NULL;
    }

    return cpuAddr;
}

void *Memory::shadow(void *addr, size_t count)
{
    TRACE(GLOBAL, "Getting shadow mapping for %p (%zd bytes)", addr, count);
    FileMapEntry entry = Files.find(addr);
    if(entry.fd() == -1) return NULL;
    off_t offset = (off_t)((uint8_t *)addr - (uint8_t *)entry.address());
    void *ret = mmap(NULL, count, ProtBits[GMAC_PROT_READWRITE], MAP_SHARED, entry.fd(), offset);
    return ret;
} 

void Memory::unshadow(void *addr, size_t count)
{
    munmap(addr, count);
}

void Memory::unmap(void *addr, size_t count)
{
    char tmp[FILENAME_MAX];
    FileMapEntry entry = Files.find(addr);
    if(Files.remove(addr) == false) return;
    munmap(addr, count);
    close(entry.fd());
}

}}

