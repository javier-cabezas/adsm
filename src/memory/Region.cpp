#include "Region.h"
#include "Manager.h"

#include <kernel/Context.h>

namespace gmac { namespace memory {

Region::Region(void *addr, size_t size, bool shared) :
    util::RWLock(LockRegion),
    _addr(__addr(addr)),
    _size(size),
    _shared(shared)
{
	_context = Context::current();
}

Region::~Region()
{
}

gmacError_t Region::copyToDevice()
{
    gmacError_t ret = gmacSuccess;
    if((ret = _context->copyToDevice(Manager::ptr(_context, start()), start(), size())) != gmacSuccess)
        return ret;
    std::list<Context *>::iterator i;
    TRACE("I have %zd relatives", _relatives.size());
    for(i = _relatives.begin(); i != _relatives.end(); i++) {
        Context * ctx = *i;
        if((ret = ctx->copyToDevice(Manager::ptr(ctx, start()), start(), size())) != gmacSuccess) {
            break;
        }
    }
    return ret;
}

gmacError_t Region::copyToHost()
{
    gmacError_t ret = gmacSuccess;
    TRACE("I have %zd relatives", _relatives.size());
    if((ret = _context->copyToHost(start(), Manager::ptr(_context, start()), size())) != gmacSuccess)
        return ret;
    return ret;
}

void Region::sync()
{
    _context->sync();
    std::list<Context *>::iterator i;
    for(i = _relatives.begin(); i != _relatives.end(); i++) {
        (*i)->sync();
    }
}

// Default implementation (only used by the Batch Memory Manager)
void Region::syncToHost()
{
    copyToHost();
}

} }
