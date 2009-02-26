#ifndef __REGION_H_
#define __REGION_H_

#include <common/config.h>

#include <stdint.h>
#include <signal.h>
#include <sys/mman.h>

#include <list>

namespace icuda {

//! Generic Memory Region Descriptor
class MemRegion {
protected:
	//! Starting memory address for the region
	void *addr;
	//! Size in bytes of the region
	size_t size;
public:
	//! Constructor
	//! \param addr Start memory address
	//! \param size Size in bytes
	MemRegion(void *addr, size_t size) : addr(addr), size(size) {}

	//! Comparision operator
	bool operator==(const void *p) const {
		return (uint8_t *)p >= (uint8_t *)addr &&
				(uint8_t *)p < ((uint8_t *)addr + size);
	}

	bool operator==(const MemRegion &m) const {
		return addr == m.addr;
	}

	//! Returns the size (in bytes) of the Region
	size_t getSize() const { return size; }
	//! Sets the size (in bytes) of the Region
	void setSize(size_t size) { this->size = size; }
	//! Returns the address of the Region
	void *getAddress() const { return addr; }
	//! Sets the address of the Region
	void setAddress(void *addr) { this->addr = addr; }
};


//! Functor to locate MemRegions inside pointer lists
class FindMem {
protected:
	void *addr;
public:
	FindMem(void *addr) : addr(addr) {};
	bool operator()(const MemRegion *r) const {
		return (*r) == addr;
	}
};


//! Functor to order MemRegions inside pointer multisets
class LessMem {
public:
	bool operator()(const MemRegion *a, const MemRegion *b) const {
		return a->getAddress() < b->getAddress();
	}
};


class ProtRegion;

//! Handler for Read/Write faults
class MemHandler {
public:
	virtual void read(ProtRegion *, void *) = 0;
	virtual void write(ProtRegion *, void *) = 0;
};


//! Protected Memory Region
class ProtRegion : public MemRegion {
protected:
	MemHandler &memHandler;
	size_t access;
	bool dirty;
	enum { None, Read, ReadWrite } permission;

	static struct sigaction defaultAction;
	static std::list<ProtRegion *> regionList;
	static void setHandler(void);
	static void restoreHandler(void);
	static void segvHandler(int, siginfo_t *, void *);
public:
	ProtRegion(MemHandler &memHandler, void *addr, size_t size);
	~ProtRegion();

	inline void read(void *addr) { memHandler.read(this, addr); }
	inline void write(void *addr) { memHandler.write(this, addr); }

	inline void noAccess(void) {
		mprotect(addr, size, PROT_NONE);
		permission = None;
	}
	inline void readOnly(void) {
		mprotect(addr, size, PROT_READ);
		permission = Read;
	}
	inline void readWrite(void) {
		mprotect(addr, size, PROT_READ | PROT_WRITE);
		permission = ReadWrite;
	}

	inline void clear() { access = 0; dirty = false; }
	inline void incAccess() { access++; }
	inline size_t getAccess() const { return access; }
	inline void setDirty() { dirty = true; }
	inline bool isDirty() const { return dirty; }
};
};

#endif
