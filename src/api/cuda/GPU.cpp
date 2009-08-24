#include "Context.h"

namespace gmac {
GPU::~GPU()
{
	std::set<gpu::Context *>::const_iterator i;
	for(i = queue.begin(); i != queue.end(); i++)
		delete *i;
	queue.clear();
}

void GPU::create()
{
	gpu::Context *ctx = new gpu::Context(*this);
	queue.insert(ctx);
}

void GPU::clone(const Context &root)
{
	gpu::Context *ctx = new gpu::Context(*this);
	queue.insert(ctx);
}

void GPU::destroy(Context *context)
{
	gpu::Context *ctx = dynamic_cast<gpu::Context *>(context);
	std::set<gpu::Context *>::iterator c = queue.find(ctx);
	assert(c != queue.end());
	queue.erase(c);
	delete *c;
}

}
