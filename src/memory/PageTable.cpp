#include "PageTable.h"

#include <util/Util.h>
#include <kernel/Context.h>

#include <malloc.h>

namespace gmac { namespace memory {

const char *PageTable::pageSizeVar = "GMAC_PAGE";
size_t PageTable::pageSize;
size_t PageTable::tableShift;

PageTable::PageTable() :
	_clean(false), _valid(true),
	pages(1), devicePages(0)
{
	MUTEX_INIT(mutex);
	const char *var = Util::getenv(pageSizeVar);
	if(var != NULL) pageSize = atoi(var);
	if(pageSize == 0) pageSize = defaultPageSize;
	tableShift = log2(pageSize);
	TRACE("Page Size: %d bytes", pageSize);
	TRACE("Table Shift: %d bits", tableShift);
	TRACE("Table Size: %d entries", (1 << dirShift) / pageSize);
}


PageTable::~PageTable()
{ 
	TRACE("Cleaning Page Table");
	for(int i = 0; i < rootTable.size(); i++) {
		if(rootTable.present(i) == false) continue;
		deleteDirectory(rootTable.value(i));
	}
}

void PageTable::deleteDirectory(Directory *dir)
{
	for(int i = 0; i < dir->size(); i++) {
		if(dir->present(i) == false) continue;
		delete dir->value(i);
	}
	delete dir;
}

void PageTable::insert(void *host, void *dev)
{
	sync();

	enterFunction(vmFlush);
	lock();
	_clean = false;
	// Get the root table entry
	if(rootTable.present(entry(host, rootShift, rootTable.size())) == false) {
		rootTable.create(entry(host, rootShift, rootTable.size()));
		pages++;
	}
	Directory &dir = rootTable.get(entry(host, rootShift, rootTable.size()));

	if(dir.present(entry(host, dirShift, dir.size())) == false) {
		dir.create(entry(host, dirShift, dir.size()),
			(1 << dirShift) / pageSize);
		pages++;
	}
	Table &table = dir.get(entry(host, dirShift, dir.size()));

	table.insert(entry(host, tableShift, table.size()), dev);
	unlock();
	exitFunction();
}

void *PageTable::translate(void *host) 
{
	sync();

	if(rootTable.present(entry(host, rootShift, rootTable.size())) == false)
		return NULL;
	Directory &dir = rootTable.get(entry(host, rootShift, rootTable.size()));
	if(dir.present(entry(host, dirShift, dir.size())) == false) return NULL;
	Table &table = dir.get(entry(host, dirShift, dir.size()));
	uint8_t *addr =
		(uint8_t *)table.value(entry(host, tableShift, table.size()));
	addr += offset(host);
	TRACE("PT translate: %p -> %p", host, addr);
	return (void *)addr;
}

bool PageTable::dirty(void *host)
{
#ifdef USE_VM
	sync();

	assert(
		rootTable.present(entry(host, rootShift, rootTable.size())) == true);
	Directory &dir = rootTable.get(entry(host, rootShift, rootTable.size()));
	assert(dir.present(entry(host, dirShift, dir.size())) == true);
	Table &table = dir.get(entry(host, dirShift, dir.size()));
	assert(table.present(entry(host, tableShift, table.size())) == true);
	return table.dirty(entry(host, tableShift, table.size()));
#else
	return true;
#endif
}

void PageTable::clear(void *host)
{
#ifdef USE_VM
	sync();
	_clean = false;
	assert(
		rootTable.present(entry(host, rootShift, rootTable.size())) == true);
	Directory &dir = rootTable.get(entry(host, rootShift, rootTable.size()));
	assert(dir.present(entry(host, dirShift, dir.size())) == true);
	Table &table = dir.get(entry(host, dirShift, dir.size()));
	assert(table.present(entry(host, tableShift, table.size())) == true);
	table.clean(entry(host, tableShift, table.size()));
#endif
}


void PageTable::flushDirectory(Directory &dir)
{
#ifdef USE_VM
	for(int i = 0; i < dir.size(); i++) {
		if(dir.present(i) == false) continue;
		TRACE("Flushing Directory entry %d", i);
		Table &table = dir.get(i);
		table.flush();
	}
	dir.flush();
#endif
}

void PageTable::syncDirectory(Directory &dir)
{
#ifdef USE_VM
	for(int i = 0; i < dir.size(); i++) {
		if(dir.present(i) == false) continue;
		TRACE("Sync Directory Entry %d", i);
		dir.get(i).sync();
	}
	dir.sync();
#endif
}

void PageTable::sync()
{
#ifdef USE_VM
		if(_valid == true) return;
		enterFunction(vmSync);
		for(int i = 0; i < rootTable.size(); i++) {
			if(rootTable.present(i) == false) continue;
			syncDirectory(rootTable.get(i));
		}
		rootTable.sync();
		_valid = true;
		exitFunction();
#endif
}


void *PageTable::flush() 
{
#ifdef USE_VM
	// If there haven't been modifications, just return
	if(_clean == true) return device;

	TRACE("PT Flush");

	for(int i = 0; i < rootTable.size(); i++) {
		if(rootTable.present(i) == false) continue;
		TRACE("Flusing entry %d", i);
		flushDirectory(rootTable.get(i));
	}

	rootTable.flush();
	_clean = true;
	return rootTable.device();
#else
	return NULL;
#endif
}

}}
