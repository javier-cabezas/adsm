#ifndef __GMAC_VM_H_
#define __GMAC_VM_H_

#include <stdint.h>

#define PAGE_TABLE_SIZE	1024
#define PAGE_SIZE 4 * 1024 * 1024
#define PAGE_SHIFT 22

__constant__ uint64_t __pageTable[PAGE_TABLE_SIZE];

__device__ inline void *__pg_lookup(void *addr)
{
	uint64_t offset = (uint64_t)addr & (PAGE_SIZE - 1);
	uint64_t base = __pageTable[((uint64_t)addr & ~(PAGE_SIZE - 1)) >> PAGE_SHIFT];

	return (void *)((base & ~(PAGE_SIZE -1)) | offset);
}

template<typename T>
__device__ inline T __globalLd(T *addr) { return *(T *)__pg_lookup(addr); }
template<typename T>
__device__ inline void __globalSt(T *addr, T v) { *(T *)__pg_lookup(addr) = v; }


#endif
