#include "Parameter.h"

#include <iostream>
#include <map>

// This CPP directives will generate parameter constructors and declare
// parameters to be used
#undef PARAM
#define PARAM(v, t, d, ...) \
    t v = d; \
    gmac::util::Parameter<t> *__##v = NULL; \
    gmac::util::__Parameter *__init__##v() { \
        __##v = new gmac::util::Parameter<t>(&v, #v, d, ##__VA_ARGS__);\
        return __##v;\
    }
#include "Parameter-def.h"

// This CPP directives will create the constructor table for all
// parameters defined by the programmer
#undef PARAM
#define PARAM(v, t, d, ...) \
    { __init__##v, __##v },
ParameterCtor ParamCtorList[] = {
#include "Parameter-def.h"
    {NULL, NULL}
};

void paramInit()
{
    for(int i = 0; ParamCtorList[i].ctor != NULL; i++)
        ParamCtorList[i].param = ParamCtorList[i].ctor();

    if(configPrintParams == true) {
        for(int i = 0; ParamCtorList[i].ctor != NULL; i++)
            ParamCtorList[i].param->print();
    }

    for(int i = 0; ParamCtorList[i].ctor != NULL; i++)
        delete ParamCtorList[i].param;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
