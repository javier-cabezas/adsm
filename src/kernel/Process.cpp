#include "Process.h"
#include "Context.h"
#include "Accelerator.h"

#include <debug.h>
#include <gmac/init.h>
#include <memory/MemManager.h>

gmac::Process *proc = NULL;

namespace gmac {
void Process::cleanAccelerators()
{
	TRACE("Cleaning accelerators");
	std::vector<Accelerator *>::iterator a;
	for(a = accs.begin(); a != accs.end(); a++)
		delete *a;
	accs.clear();
}

Process::~Process()
{
	TRACE("Cleaning process");
	cleanAccelerators();
	memoryFini();
}

void Process::create()
{
	TRACE("Creating new context");
	accs[current]->create();
	current = ++current % accs.size();
}

void Process::clone()
{
	TRACE("Cleaning process");
	cleanAccelerators();
	manager->clean();

	TRACE("Restarting process");
	apiInit();
	apiInitDevices();
	create();
	gmac::Context::current()->clone();
}
}
