#include "Region.h"
#include "Manager.h"

#include <kernel/Context.h>

namespace gmac { namespace memory {

Region::Region(void *addr, size_t size) :
   _lock(paraver::relatives),
	_addr(__addr(addr)),
	_size(size)
{
	_context = Context::current();
}

Region::~Region()
{
}

gmacError_t Region::copyToDevice()
{
    gmacError_t ret = gmacSuccess;
    if((ret = _context->copyToDevice(Manager::ptr(start()), start(), size())) != gmacSuccess)
        return ret;
    std::list<Context *>::iterator i;
    _lock.read();
    for(i = _relatives.begin(); i != _relatives.end(); i++) {
        if((ret = (*i)->copyToDevice(Manager::ptr(start()), start(), size())) != gmacSuccess) {
            _lock.unlock();
            return ret;
        }
    }
    _lock.unlock();
    return ret;	
}

gmacError_t Region::copyToHost()
{
	gmacError_t ret = gmacSuccess;
	if((ret = _context->copyToHost(start(), Manager::ptr(start()), size())) != gmacSuccess)
		return ret;
	return ret;	
}

void Region::sync()
{
	_context->sync();
	std::list<Context *>::iterator i;
   _lock.read();
	for(i = _relatives.begin(); i != _relatives.end(); i++) {
		(*i)->sync();
	}
   _lock.unlock();
}

} }
