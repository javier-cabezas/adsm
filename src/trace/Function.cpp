#include "Function.h"
#include <util/Logger.h>

#include <ostream>
#include <cassert>

namespace gmac { namespace trace {

#ifdef PARAVER
unsigned FunctionMap::_count = 0;
const char *Function::eventName = "Function";
ModuleMap *Function::map = NULL;
paraver::EventName *Function::event = NULL;
#endif

void Function::init()
{
#ifdef PARAVER
    map = new ModuleMap();
    event = paraver::Factory<paraver::EventName>::create(eventName);
#endif
}


void Function::start(const char *module, const char *name)
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    map->lock();
    FunctionMap &function = (*map)[std::string(module)];

    FunctionMap::const_iterator i = function.find(std::string(name));
    unsigned id = -1;
    if(i == function.end()) {
        id = function.id() + function.size() + 1;
        std::ostringstream os;
        os << module << "::" << name;
        event->registerType(id, os.str());
        function.insert(FunctionMap::value_type(std::string(name), id));
    }
    else id = i->second;
    map->unlock();
    paraver::trace->__pushEvent(*event, id);
#endif
}

void Function::end()
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    util::Logger::ASSERTION(event != NULL);

    paraver::trace->__pushEvent(*event, 0);
#endif
}

}}
