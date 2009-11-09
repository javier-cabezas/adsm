#include "Region.h"
#include "Manager.h"

#include <kernel/Context.h>

namespace gmac { namespace memory {

Region::Region(void *addr, size_t size) :
	_addr(__addr(addr)),
	_size(size)
{
	_context = Context::current();
}

gmacError_t Region::copyToDevice()
{
	return _context->copyToDevice(Manager::ptr(start()), start(), size());
}

gmacError_t Region::copyToHost()
{
	return _context->copyToHost(start(), Manager::ptr(start()), size());
}

void Region::sync()
{
	_context->sync();
}

} }
