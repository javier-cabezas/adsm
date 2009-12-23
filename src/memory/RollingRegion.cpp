#include "RollingRegion.h"
#include "RollingManager.h"

#include "os/Memory.h"

#include <algorithm>

namespace gmac { namespace memory {



RollingRegion::RollingRegion(RollingManager &manager, void *addr, size_t size,
		size_t cacheLine) :
	Region(addr, size),
	manager(manager),
	cacheLine(cacheLine),
	offset((unsigned long)addr & (cacheLine -1))
{
	TRACE("RollingRegion Starts");
	for(size_t s = 0; s < size; s += cacheLine) {
		void *p = (void *)((uint8_t *)addr + s);
		size_t regionSize = ((size -s) > cacheLine) ? cacheLine : (size - s);
		ProtSubRegion *region = new ProtSubRegion(this, p, regionSize);
		void *key = (void *)((uint8_t *)p + cacheLine);
		map.insert(Map::value_type(key, region));
		memory.insert(region);
	}
	TRACE("RollingRegion Ends");
}

RollingRegion::~RollingRegion()
{
	Map::const_iterator i;
	for(i = map.begin(); i != map.end(); i++) {
		manager.invalidate(i->second);
		delete i->second;
	}
	map.clear();
}

void RollingRegion::relate(Context *ctx)
{
	Map::const_iterator i;
	// Push dirty regions in the rolling buffer
	// and copy to device clean regions
	for(i = map.begin(); i != map.end(); i++) {
		if(i->second->dirty()) {
            if (!manager.regionRolling[ctx]) {
                manager.regionRolling[ctx] = new RollingBuffer();
            }
            manager.regionRolling[ctx]->push(i->second);
        } else assert(ctx->copyToDevice(Manager::ptr(start()), start(), size()) == gmacSuccess);
		i->second->relate(ctx);
	}
	_relatives.push_back(ctx);
	manager.regionRolling[ctx]->inc(manager.lruDelta);
}

void RollingRegion::unrelate(Context *ctx)
{
	Map::iterator i;
	for(i = map.begin(); i != map.end(); i++) i->second->unrelate(ctx);
	_relatives.remove(ctx);
}

void RollingRegion::transfer()
{
	Map::iterator i;
	for(i = map.begin(); i != map.end(); i++) i->second->transfer();
	Region::transfer();
}

ProtSubRegion *RollingRegion::find(const void *addr)
{
	Map::const_iterator i = map.upper_bound(addr);
	if(i == map.end()) return NULL;
	if((addr_t)addr < (addr_t)i->second->start()) return NULL;
	return i->second;
}


void RollingRegion::invalidate()
{
	TRACE("RollingRegion Invalidate %p (%d bytes)", _addr, _size);
	// Check if the region is already invalid
	if(memory.empty()) return;

	// Protect the region
	Memory::protect(__void(_addr), _size, PROT_NONE);
	// Invalidate those sub-regions that are present in memory
	List::iterator i;
	for(i = memory.begin(); i != memory.end(); i++) {
		TRACE("Invalidate SubRegion %p (%d bytes)", (*i)->start(),
				(*i)->size());
		(*i)->silentInvalidate();
	}
	memory.clear();
}

void RollingRegion::invalidate(const void *addr, size_t size)
{
	void *end = (void *)((addr_t)addr + size);
	Map::iterator i = map.lower_bound(addr);
	assert(i != map.end());
	for(; i != map.end() && i->second->start() < end; i++) {
		// If the region is not present, just ignore it
		if(i->second->present() == false) continue;

		if(i->second->dirty()) { 	// We might need to update the device
			// Check if there is memory that will not be invalidated
			if(i->second->start() < addr ||
					i->second->end() > __void(__addr(addr) + size))
				manager.flush(i->second);
		}
		memory.erase(i->second);
		i->second->invalidate();
	}
}

void RollingRegion::flush(const void *addr, size_t size)
{
	Map::iterator i = map.lower_bound(addr);
	assert(i != map.end());
	for(; i != map.end() && i->second->start() < addr; i++) {
		// If the region is not present, just ignore it
		if(i->second->dirty() == false) continue;
		manager.flush(i->second);
		i->second->readOnly();
	}
}

ProtSubRegion::ProtSubRegion(RollingRegion *parent, void *addr, size_t size) :
		ProtRegion(addr, size),
		parent(parent)
{ }


ProtSubRegion::~ProtSubRegion()
{
    TRACE("SubRegion %p released", _addr);
}

void
ProtSubRegion::readOnly()
{
    if(present() == false) parent->push(this);
    ProtRegion::readOnly();
}

void
ProtSubRegion::readWrite()
{
    if(present() == false) parent->push(this);
    ProtRegion::readWrite();
}

} };
