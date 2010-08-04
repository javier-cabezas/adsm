#include "Allocator.h"

#include "allocator/Slab.h"

namespace gmac { namespace memory {

int Allocator::__count = 0;
Allocator *Allocator::__allocator = NULL;

Allocator *Allocator::create()
{
    __count++;
    if(__allocator != NULL) return __allocator;
    gmac::util::Logger::TRACE("Creating Memory Allocator");
    __allocator = new allocator::Slab();
    return __allocator;
}

void Allocator::destroy()
{
    __count--;
    if(__count > 0) return;
    gmac::util::Logger::TRACE("Destroying Memory Allocator");
    delete __allocator;
    __allocator = NULL;
}

Allocator *Allocator::get()
{
    gmac::util::Logger::ASSERTION(__allocator != NULL);
    return __allocator;
}

}}
