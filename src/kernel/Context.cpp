#include "Context.h"

#include <threads.h>


namespace gmac {

void contextInit()
{
	PRIVATE_INIT(gmac::Context::key, NULL);
	PRIVATE_SET(gmac::Context::key, NULL);
}


PRIVATE(Context::key);

std::list<Context *> *Context::list = NULL;
}
