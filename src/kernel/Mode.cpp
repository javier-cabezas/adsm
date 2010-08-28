#include "Mode.h"
#include "Accelerator.h"

#include <memory/Object.h>
#include <memory/Map.h>

namespace gmac {

gmac::util::Private Mode::key;
unsigned Mode::next = 0;

Mode::Mode(Accelerator *acc) :
    __id(next++),
    __acc(acc)
{
    trace("Creating new context");
    __context = createContext();
    trace("Creating new memory map");
    __map = new memory::Map(paraver::LockMmLocal);
}

Mode::~Mode()
{
    __count--;
    if(__count > 0)
        gmac::util::Logger::WARNING("Deleting in-use Execution Mode");
    if(this == key.get()) key.set(NULL);
    delete __map;
    delete __context;
    __acc->destroyMode(this); 
}


}
