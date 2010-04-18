#include "ProtRegion.h"

namespace gmac {
namespace memory {

ProtRegion::ProtRegion(void *addr, size_t size, bool shared) :
    Region(addr, size, shared),
    _dirty(false),
    _present(true)
{}

ProtRegion::~ProtRegion()
{}

void ProtRegion::syncToHost()
{
    if (_present == false) {
        readWrite();
        copyToHost();
        readOnly();
    }
}

}}
