#include "Context.h"

#include <threads.h>


namespace gmac {

void contextInit()
{
	PRIVATE_INIT(gmac::Context::key, NULL);
}


PRIVATE(Context::key);
}
