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

#include "params.h"

#include <iostream>
#include <map>

namespace gmac { namespace params {

std::vector<Root *> *Root::__params = NULL;

Root::Root(const char *name, const char *envVar, uint32_t flags) :
    name(name), envVar(envVar), flags(flags), envSet(false)
{ 
    TRACE("Getting value for %s", name);
    params().push_back(this);
}

void Root::print()
{
    std::cout << name << std::endl;
    value(std::string("\tValue: "), std::cout); std::cout << std::endl;
    def(std::string("\tDefault: "), std::cout); std::cout << std::endl;
    std::cout << "\tVariable: " << envVar << std::endl;
    std::cout << "\tFlags: " << flags << std::endl;
    std::cout << "\tSet: " << envSet << std::endl;
}
} }

PARAM_REGISTER(configPrintParams,
               bool,
               false,
               "GMAC_PRINT_PARAMS");

void paramInit()
{
    if(configPrintParams == true) {
        std::vector<gmac::params::Root *>::const_iterator i;
        for(i = gmac::params::Root::params().begin();
                i != gmac::params::Root::params().end(); i++)
            (*i)->print();
    }
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
