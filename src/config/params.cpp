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

#include "params.h"

#include <iostream>
#include <map>

std::map<const char *, ParamDescriptor> params __attribute__((init_priority(CONFIG - 1)));

PARAM_REGISTER(configPrintParams,
               bool,
               false,
               "GMAC_PRINT_PARAMS");

static
void __attribute__((constructor(DEFAULT))) doPrintParams()
{
    if (configPrintParams) {
        std::ios_base::Init i;
        for (std::map<const char *, ParamDescriptor>::iterator it = params.begin();
                it != params.end();
                it++) {
            
            std::cout << it->first << std::endl;
            std::cout << "\tvalue:   ";
            it->second.print();
            std::cout << std::endl;
            std::cout << "\tdefault: ";
            it->second.print_default();
            std::cout << std::endl;
            if (it->second.env != NULL) {
                std::cout << "\tenv_var: " << it->second.env   << std::endl;
            } else {
                std::cout << "\tenv_var: (none)" << std::endl;
            }
            std::cout << "\tflags:   " << it->second.flags << std::endl;
            if (it->second.envSet) {
                std::cout << "\tset:     true" << std::endl;
            } else {
                std::cout << "\tset:     false" << std::endl;
            }
        }
    }
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
