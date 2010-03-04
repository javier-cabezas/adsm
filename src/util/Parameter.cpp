/*
 * <+ DESCRIPTION +>
 *
 * Copyright (C) 2009, Javier Cabezas <jcabezas in ac upc edu> {{{
 *
 * This program is free software; you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License 
 * as published by the Free Software Foundation; either 
 * version 2 of the License, or any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 * }}}
 */

#include <debug.h>

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
#include "Parameter.def"

// This CPP directives will create the constructor table for all
// parameters defined by the programmer
#undef PARAM
#define PARAM(v, t, d, ...) \
    { __init__##v, __##v },
ParameterCtor ParamCtorList[] = {
#include "Parameter.def"
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
