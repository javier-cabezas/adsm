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
	lock();
	for(a = accs.begin(); a != accs.end(); a++)
		delete *a;
	accs.clear();
	unlock();
	memoryFini();
}

void Process::create()
{
	TRACE("Creating new context");
	lock();
	unsigned n = current;
	current = ++current % accs.size();
	unlock();
	accs[n]->create();
}

void Process::clone(const gmac::Context *ctx)
{
	TRACE("Cloning context");
	lock();
	unsigned n = current;
	current = ++current % accs.size();
	unlock();
	accs[n]->clone(*ctx);
}

}
