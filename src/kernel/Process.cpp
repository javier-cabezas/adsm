#include "Process.h"
#include "Context.h"
#include "Accelerator.h"

#include <debug.h>
#include <gmac/init.h>
#include <memory/MemManager.h>

gmac::Process *proc = NULL;

namespace gmac {

Process::~Process()
{
	TRACE("Cleaning process");
	std::vector<Accelerator *>::iterator a;
	MUTEX_LOCK(mutex);
	for(a = accs.begin(); a != accs.end(); a++)
		delete *a;
	accs.clear();
	MUTEX_UNLOCK(mutex);
	memoryFini();
}

void Process::create()
{
	TRACE("Creating new context");
	MUTEX_LOCK(mutex);
	accs[current]->create();
	current = ++current % accs.size();
	MUTEX_UNLOCK(mutex);
}

void Process::clone(const gmac::Context *ctx)
{
	TRACE("Cloning context");
	MUTEX_LOCK(mutex);
	accs[current]->clone(*ctx);
	current = ++current % accs.size();
	MUTEX_UNLOCK(mutex);
}

}
