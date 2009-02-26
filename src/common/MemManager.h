#ifndef __MEMMANAGER_H_
#define __MEMMANAGER_H_

#include <common/config.h>

#include <stdint.h>
#include <sys/mman.h>

namespace icuda {
//! Memory Manager Interface

//! Memory Managers implement a policy to move data from/to
//! the CPU memory to/from the GPU memory.
class MemManager {
protected:
	//! This method maps a GPU address into the CPU address space
	//! \param addr GPU address
	//! \param count Size (in bytes) of the mapping
	//! \param prot Protection flags for the mapping
	void *map(void *addr, size_t count, int prot = PROT_READ | PROT_WRITE);

	//! This method upmaps a GPU address from the CPU address space
	//! \param addr GPU address
	//! \param count Size (in bytes) to unmap
	void unmap(void *addr, size_t count);

public:
	//! Virtual Destructor. It does nothing
	virtual ~MemManager() {}

	//! This method is called whenever the user
	//! requests memory to be used by the GPU
	//! \param devPtr Allocated memory address. This address
	//! is the same for both, the CPU and the GPU.
	//! \param count Size in bytes of the allocated memory
	virtual bool alloc(void *addr, size_t count) = 0;

	//! This method is called whenever the user
	//! releases GPU memory
	//! \param devPtr Memory address that has been released
	virtual void release(void *addr) = 0;

	//! This method is called whenever the user invokes
	//! a kernel to be executed at the GPU
	virtual void execute(void) = 0;

	//! This method is called just after the user requests
	//! waiting for the GPU to finish
	virtual void sync(void) = 0;
};


//! Gets a Memory Manager based on a string name
//! \param managerName Name of the memory manager
MemManager *getManager(const char *managerName);

};
#endif
