#include "ProtRegion.h"

namespace gmac {
namespace memory {

ProtRegion::ProtRegion(void *addr, size_t size) :
    Region(addr, size),
    _dirty(false),
    _present(true)
{}

ProtRegion::~ProtRegion()
{}

}}
