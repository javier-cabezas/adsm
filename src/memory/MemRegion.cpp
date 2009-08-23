#include "MemRegion.h"

#include <kernel/Context.h>

namespace gmac {

MemRegion::MemRegion(void *addr, size_t size) :
	_addr(__addr(addr)),
	_size(size)
{
	_context = Context::current();
}

}
