#include "MemMap.h"

namespace gmac {

MUTEX(MemMap::global);
MemMap::Map MemMap::__global;
}
