#include <os/loader.h>
#include <kernel/Process.h>
#include <kernel/Context.h>

#include <order.h>
#include <paraver.h>
#include <debug.h>

#include <unistd.h>
#include <dlfcn.h>

#include <errno.h>

#include "posix.h"

void posixForkInit(void)
{
	TRACE("Overloading POSIX fork");
}


#ifdef __cplusplus
extern "C" {
#endif

pid_t fork()
{
	FATAL("fork() not supported by GMAC");
}

int clone(int (*fn)(void *), void *, int, void *, ...)
{
	FATAL("clone() not supported by GMAC");
	return 0;
}

#ifdef __cplusplus
}
#endif
