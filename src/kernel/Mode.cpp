#include "Mode.h"
#include "Accelerator.h"

#include <memory/Object.h>
#include <memory/Map.h>

namespace gmac {

gmac::util::Private Mode::key;
unsigned Mode::next = 0;

Mode::Mode(Accelerator *acc) :
    __id(next++),
    acc(acc)
{
    trace("Creating new memory map");
    map = new memory::Map(paraver::LockMmLocal);
}

Mode::~Mode()
{
    count--;
    if(count > 0)
        gmac::util::Logger::WARNING("Deleting in-use Execution Mode");
    if(this == key.get()) key.set(NULL);
    delete map;
    delete context;
    acc->destroyMode(this); 
}


}
