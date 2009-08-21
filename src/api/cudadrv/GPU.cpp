#include "GPU.h"
#include "Context.h"

#include <debug.h>


namespace gmac {

GPU::~GPU()
{
	std::set<gpu::Context *>::const_iterator i;
	for(i = runQueue.begin(); i != runQueue.end(); i++)
		delete *i;
	runQueue.clear();
}

Context *GPU::create()
{
	gpu::Context *ctx = new gpu::Context(*this);
	runQueue.insert(ctx);
	return ctx;
}

void GPU::destroy(Context *context)
{
	gpu::Context *ctx = dynamic_cast<gpu::Context *>(context);
	std::set<gpu::Context *>::iterator c = runQueue.find(ctx);
	assert(c != runQueue.end());
	runQueue.erase(c);
	delete ctx;
}

};
