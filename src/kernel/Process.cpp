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
	for(a = accs.begin(); a != accs.end(); a++)
		delete *a;
	accs.clear();
	memoryFini();
}

void Process::context()
{
	TRACE("Creating new context");
	accs[current]->create();
	current = ++current % accs.size();
}

}
