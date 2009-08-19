#include "Context.h"

#include <threads.h>


namespace gmac {

static void destroyContext(void *c)
{
	gmac::Context *context = static_cast<gmac::Context *>(c);
	delete context;
}

void contextInit()
{
	PRIVATE_INIT(gmac::Context::key, destroyContext);
}


PRIVATE(Context::key);
}
