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

static unsigned n = 0;
static FileMap Files;

int Memory::protect(void *addr, size_t count, Protection prot)
{
    util::Logger::TRACE("Setting memory permisions to %d @ %p - %p", prot, addr, (uint8_t *)addr + count);
    int ret = mprotect(addr, count, ProtBits[prot]);
    util::Logger::ASSERTION(ret == 0);
    return 0;
}

void *Memory::map(void *addr, size_t count, Protection prot)
{
    void *cpuAddr = NULL;
    char tmp[FILENAME_MAX];

    // Create new shared memory file
    int fd = -1;
    unsigned id = 0;
    while(fd < 0) {
        id = n++;
        snprintf(tmp, FILENAME_MAX, "/gmac%d", id);
        fd = shm_open(tmp, O_RDWR | O_CREAT | O_EXCL, S_IRWXU | S_IRWXG);
        if(errno != EEXIST) return NULL;
    }

    if (addr == NULL) {
        cpuAddr = mmap(addr, count, ProtBits[prot], MAP_PRIVATE, fd, 0);
        util::Logger::TRACE("Getting map: %d @ %p - %p", prot, cpuAddr, (uint8_t *)addr + count);
    } else {
        cpuAddr = addr;
        if(mmap(cpuAddr, count, ProtBits[prot], MAP_PRIVATE | MAP_FIXED, fd, 0) != cpuAddr) {
            close(fd);
            shm_unlink(tmp);
            return NULL;
        }
        util::Logger::TRACE("Getting fixed map: %d @ %p - %p", prot, addr, (uint8_t *)addr + count);
    }

    if(Files.insert(fd, cpuAddr, count, id) == false) {
        munmap(cpuAddr, count);
        close(fd);
        shm_unlink(tmp);
        return NULL;
    }

    return cpuAddr;
}

void *Memory::shadow(void *addr, size_t count)
{
    util::Logger::TRACE("Getting shadow mapping for %p (%zd bytes)", addr, count);
    FileMapEntry entry = Files.find(addr);
    if(entry.fd() == -1) return NULL;
    off_t offset = (off_t)((uint8_t *)addr - (uint8_t *)entry.address());
    void *ret = mmap(NULL, count, ProtBits[ReadWrite], MAP_PRIVATE, entry.fd(), offset);
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
    snprintf(tmp, FILENAME_MAX, "/gmac%d", entry.id());
    shm_unlink(tmp);
}

}}

