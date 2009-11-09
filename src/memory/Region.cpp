#include "Region.h"

#include <kernel/Context.h>

namespace gmac { namespace memory {

Region::Region(void *addr, size_t size) :
	_addr(__addr(addr)),
	_size(size)
{
	_context = Context::current();
}

} }
