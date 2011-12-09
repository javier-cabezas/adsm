#include "memory/windows/FileMap.h"

namespace __impl { namespace memory {

map_file::map_file() :
	gmac::util::lock_rw("FileMap")
{ }

map_file::~map_file()
{ }

bool map_file::insert(HANDLE handle, hostptr_t address, size_t size)
{
	hostptr_t key = address + size;
	lock_write();
	std::pair<Parent::iterator, bool> ret = Parent::insert(
		Parent::value_type(key, map_file_entry(handle, address, size)));
	unlock();
	return ret.second;
}

bool map_file::remove(hostptr_t address)
{
	bool ret = true;
	lock_write();
	Parent::const_iterator i = Parent::upper_bound(address);
	if(i != Parent::end()) Parent::erase(i);
	else ret = false;
	unlock();
	return ret;
}

const map_file_entry map_file::find(hostptr_t address) const
{
	map_file_entry ret(NULL, NULL, 0);
	lock_read();
	Parent::const_iterator i = Parent::upper_bound(address);
	if(i != Parent::end()) {
		if((uint8_t *)i->second.address() <= (uint8_t *) address) {
			ret = i->second;
		}
	}
	unlock();
	return ret;
}
}}

