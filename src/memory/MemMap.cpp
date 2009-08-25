#include "MemMap.h"

namespace gmac {


MemMap::Map *MemMap::__global = NULL;
unsigned MemMap::count = 0;
MUTEX(MemMap::global);
}
