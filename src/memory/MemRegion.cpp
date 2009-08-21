#include "MemRegion.h"

#include <kernel/Context.h>

namespace gmac {

MemRegion::MemRegion(void *addr, size_t size) :
	addr(__addr(addr)),
	size(size)
{
	_context = Context::current();
}

}
