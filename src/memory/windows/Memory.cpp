#include "memory/Memory.h"
#include "memory/windows/FileMap.h"
#include "core/Mode.h"

namespace gmac { namespace memory {

static DWORD ProtBits[] = {
	PAGE_NOACCESS,
	PAGE_READONLY,
	PAGE_READWRITE,
	PAGE_READWRITE
};


static FileMap Files;

int Memory::protect(void *addr, size_t count, Protection prot)
{
    util::Logger::TRACE("Setting memory permisions to %d @ %p - %p", prot, addr, (uint8_t *)addr + count);
	DWORD old = 0;
    BOOL ret = VirtualProtect(addr, count, ProtBits[prot], &old);
    util::Logger::ASSERTION(ret == TRUE);
    return 0;
}

void *Memory::map(void *addr, size_t count, Protection prot)
{
    void *cpuAddr = NULL;

#if 0
    if (addr == NULL) {
        cpuAddr = VirtualAlloc(NULL, count, MEM_RESERVE, ProtBits[prot]);
        util::Logger::TRACE("Getting map: %d @ %p - %p", prot, cpuAddr, (uint8_t *)addr + count);
    } else {
        cpuAddr = addr;
		if(VirtualAlloc(cpuAddr, count, MEM_RESERVE, ProtBits[prot]) != cpuAddr)
			return NULL;
        util::Logger::TRACE("Getting fixed map: %d @ %p - %p", prot, addr, (uint8_t *)addr + count);
    }
#endif

	// Create anonymous file to be mapped
	HANDLE handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)count, NULL);
	if(handle == NULL) return NULL;
	
	// Create a view of the file in the previously allocated virtual memory range
	if((cpuAddr = MapViewOfFileEx(handle, FILE_MAP_WRITE, 0, 0, (DWORD)count, addr)) == NULL) {
		CloseHandle(handle);
		return NULL;
	}

	if(Files.insert(handle, cpuAddr, count) == false) {
		CloseHandle(handle);
		return NULL;
	}

	// Make sure that the range has the requested virtual protection bits
	protect(cpuAddr, count, prot);
    return cpuAddr;
}

void *Memory::remap(void *addr, void *to, size_t count, Protection prot)
{
    util::Logger::TRACE("Getting fixed remap: %d @ %p -> %p", prot, addr, to);
	FileMapEntry entry = Files.find(to);
	if(entry.handle() == NULL) return NULL;
	off_t offset = (off_t)((uint8_t *)to - (uint8_t *)entry.address());
	void *tmp = MapViewOfFile(entry.handle(), FILE_MAP_WRITE, 0, (DWORD)offset, (DWORD)count);
	if(tmp == NULL) return NULL;
	memcpy(tmp, addr, count);
	BOOL ret = UnmapViewOfFile(tmp);
	util::Logger::ASSERTION(ret == TRUE);
	DWORD dummy;
	ret = VirtualProtect(to, count, ProtBits[prot], &dummy);
	util::Logger::ASSERTION(ret == TRUE);
	return to;
}

void Memory::unmap(void *addr, size_t /*count*/)
{
	FileMapEntry entry = Files.find(addr);
	if(Files.remove(addr) == false) return;
	UnmapViewOfFile(entry.address());
	CloseHandle(entry.handle());
}

}}

