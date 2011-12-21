#include "memory/memory.h"
#include "memory/windows/map_file.h"
#include "core/Mode.h"

namespace __impl { namespace memory {

static DWORD ProtBits[] = {
	PAGE_NOACCESS,
	PAGE_READONLY,
	PAGE_READWRITE,
	PAGE_READWRITE
};


static map_file Files;

int memory_ops::protect(host_ptr addr, size_t count, GmacProtection prot)
{
    TRACE(GLOBAL, "Setting memory permisions to %d @ %p - %p", prot, addr, addr + count);
	DWORD old = 0;
    BOOL ret = VirtualProtect(addr, count, ProtBits[prot], &old);
    ASSERTION(ret == TRUE);
    return 0;
}

host_ptr memory_ops::map(host_ptr addr, size_t count, GmacProtection prot)
{
    host_ptr cpuAddr = NULL;

	// Create anonymous file to be mapped
	HANDLE handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)count, NULL);
	if(handle == NULL) return NULL;
	
	// Create a view of the file in the previously allocated virtual memory range
	if((cpuAddr = (host_ptr)MapViewOfFileEx(handle, FILE_MAP_WRITE, 0, 0, (DWORD)count, addr)) == NULL) {
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

host_ptr memory_ops::shadow(host_ptr addr, size_t count)
{
	TRACE(GLOBAL, "Getting shadow mapping for %p ("FMT_SIZE" bytes)", addr, count);
	map_file_entry entry = Files.find(addr);
	if(entry.handle() == NULL) return NULL;
	off_t offset = off_t(addr - entry.address());
	host_ptr ret = (host_ptr)MapViewOfFile(entry.handle(), FILE_MAP_WRITE, 0, (DWORD)offset, (DWORD)count);
	return ret;
}

void memory_ops::unshadow(host_ptr addr, size_t /*count*/)
{
	BOOL ret = UnmapViewOfFile(addr);
	ASSERTION(ret == TRUE);
}


void memory_ops::unmap(host_ptr addr, size_t /*count*/)
{
	map_file_entry entry = Files.find(addr);
	if(Files.remove(addr) == false) return;
	UnmapViewOfFile(entry.address());
	CloseHandle(entry.handle());
}

}}

