#include "MemHandler.h"

#include <memory/ProtRegion.h>

namespace gmac {
MemHandler *MemHandler::handler = NULL;


ProtRegion *MemHandler::find(void *addr) {
	return mm.find<ProtRegion>(addr);
}
};
