#include "Process.h"
#include "Context.h"
#include "Accelerator.h"

#include <debug.h>
#include <gmac/init.h>
#include <memory/Manager.h>

gmac::Process *proc = NULL;

namespace gmac {

size_t Process::_totalMemory = 0;

Process::~Process()
{
	TRACE("Cleaning process");
	std::vector<Accelerator *>::iterator a;
	std::list<Context *>::iterator c;
	lock();
	for(c = contexts.begin(); c != contexts.end(); c++) {
		(*c)->destroy();
	}
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
	contexts.push_back(accs[n]->create());
	contexts.back()->init();
}

void Process::clone(gmac::Context *ctx)
{
	TRACE("Cloning context");
	lock();
	unsigned n = current;
	current = ++current % accs.size();
	unlock();
	contexts.push_back(accs[n]->clone(*ctx));
	contexts.back()->init();
	TRACE("Cloned context on Acc#%d", n);
}

void Process::remove(Context *ctx)
{
	contexts.remove(ctx);
	ctx->destroy();
}

void Process::accelerator(Accelerator *acc) 
{
	accs.push_back(acc);
	_totalMemory += acc->memory();
}

void *Process::translate(void *addr) 
{
	void *ret = NULL;
	std::list<Context *>::const_iterator i;
	for(i = contexts.begin(); i != contexts.end(); i++) {
		ret = (*i)->mm().pageTable().translate(addr);
		if(ret != NULL) return ret;
	}
	return ret;
}


}
