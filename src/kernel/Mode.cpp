#include "Mode.h"
#include "IOBuffer.h"
#include "Accelerator.h"

#include <memory/Object.h>
#include <memory/Map.h>
#include <memory/Manager.h>
#include <gmac/init.h>

namespace gmac {

gmac::util::Private Mode::key;
unsigned Mode::next = 0;

Mode::Mode(Accelerator *acc) :
    __id(++next),
    acc(acc),
    count(0)
{
    trace("Creating new memory map");
    map = new memory::Map(paraver::LockMmLocal);
}

Mode::~Mode()
{
    count--;
    if(count > 0)
        gmac::util::Logger::WARNING("Deleting in-use Execution Mode (%d)", count);
    if(this == key.get()) key.set(NULL);

    delete map;
    acc->destroyMode(this); 
}


bool Mode::requireUpdate(memory::Block *block)
{
    return manager->requireUpdate(block);
}

}
