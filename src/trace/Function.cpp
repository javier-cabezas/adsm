#include "Function.h"
#include <util/Logger.h>

#include <ostream>
#include <cassert>

namespace gmac { namespace trace {

#ifdef PARAVER
ModuleMap *Function::map = NULL;

FunctionMap::FunctionMap(unsigned n, const char *name)
{
    _event = paraver::Factory<paraver::EventName>::create(name);
    id_ = n * _stride;
}


FunctionMap &ModuleMap::get(const char *module)
{
    FunctionMap *func = NULL;
    iterator i = find(std::string(module));
    if(i != end())  func = i->second;
    else {
        func = new FunctionMap(size(), module);
        insert(value_type(std::string(module), func));
    }
    return *func;
}

#endif



void Function::init()
{
#ifdef PARAVER
    map = new ModuleMap();
#endif
}


void Function::start(const char *module, const char *name)
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    map->lock();
    FunctionMap &function = map->get(module);

    FunctionMap::const_iterator i = function.find(std::string(name));
    unsigned id = -1;
    if(i == function.end()) {
        id = function.id() + function.size() + 1;
        function.event().registerType(id, std::string(name));
        function.insert(FunctionMap::value_type(std::string(name), id));
    }
    else id = i->second;
    map->unlock();
    paraver::trace->__pushEvent(function.event(), id);
#endif
}

void Function::end(const char *module)
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;

    FunctionMap &function = map->get(module);
    paraver::trace->__pushEvent(function.event(), 0);
#endif
}

}}
