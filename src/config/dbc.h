/*
 * Copyright (C) 2010 Lluís Vilanova <vilanova@ac.upc.edu>
 *
 * This file is part of libparatrace.
 *
 * Libparatrace is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#ifndef GMAC_CONFIG_DBC_H_
#define GMAC_CONFIG_DBC_H_

namespace gmac {
    namespace memory {

        namespace protocol {
            namespace __dbc { }
            using namespace __dbc;
        }
        namespace __dbc { 
            class Block;
            class Manager;
        }

        using namespace __dbc;
    }

    namespace util {
        namespace __dbc { }
        using namespace __dbc;
    }
}

#endif  /* GMAC_CONFIG_VISIBILITY_H */
