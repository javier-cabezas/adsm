#include "memory/posix/FileMap.h"

namespace gmac { namespace memory {


FileMapEntry::FileMapEntry(int fd, void *address, size_t size, const char *name) :
	fd_(fd), address_(address), size_(size)
{
    if(name == NULL) { name_ = NULL; return; }
    name_ = new char[strlen(name) + 1];
    memcpy(name_, name, strlen(name) + 1);
}

FileMapEntry::FileMapEntry(const FileMapEntry &e) :
    fd_(e.fd_), address_(e.address_), size_(e.size_)
{
    if(e.name_ == NULL) return;
    name_ = new char[strlen(e.name_) + 1];
    memcpy(name_, e.name_, strlen(e.name_) + 1);
}

FileMapEntry::~FileMapEntry()
{
    if(name_ != NULL) delete name_;
}


FileMapEntry & FileMapEntry::operator=(const FileMapEntry &e)
{
    if(this == &e) return *this;
    fd_ = e.fd_;
    address_ = e.address_;
    size_ = e.size_;
    if(e.name_ == NULL) return *this;
    name_ = new char[strlen(e.name_) + 1];
    memcpy(name_, e.name_, strlen(e.name_) + 1);
    return *this;
}

FileMap::FileMap() :
	util::RWLock("FileMap")
{ }

FileMap::~FileMap()
{ }

bool FileMap::insert(int fd, void *address, size_t size, const char *name)
{
	void *key = (uint8_t *)address + size;
	lockWrite();
	std::pair<Parent::iterator, bool> ret = Parent::insert(
		Parent::value_type(key, FileMapEntry(fd, address, size, name)));
	unlock();
	return ret.second;
}

bool FileMap::remove(void *address)
{
	bool ret = true;
	lockWrite();
	Parent::iterator i = Parent::upper_bound(address);
	if(i != Parent::end()) Parent::erase(i);
	else ret = false;
	unlock();
	return ret;
}

const FileMapEntry FileMap::find(void *address) const
{
	FileMapEntry ret(-1, NULL, 0, 0);
	lockRead();
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

