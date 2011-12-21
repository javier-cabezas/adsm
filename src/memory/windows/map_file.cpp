#include "memory/windows/map_file.h"

namespace __impl { namespace memory {

map_file::map_file() :
	gmac::util::lock_rw("map_file")
{ }

map_file::~map_file()
{ }

bool map_file::insert(HANDLE handle, host_ptr address, size_t size)
{
	host_ptr key = address + size;
	lock_write();
	std::pair<Parent::iterator, bool> ret = Parent::insert(
		Parent::value_type(key, map_file_entry(handle, address, size)));
	unlock();
	return ret.second;
}

bool map_file::remove(host_ptr address)
{
	bool ret = true;
	lock_write();
	Parent::const_iterator i = Parent::upper_bound(address);
	if(i != Parent::end()) Parent::erase(i);
	else ret = false;
	unlock();
	return ret;
}

const map_file_entry map_file::find(host_ptr address) const
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

