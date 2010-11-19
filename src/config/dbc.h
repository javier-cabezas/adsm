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

#if 0
namespace gmac {
    namespace __impl {
        namespace memory {
            class Block;

            namespace protocol {
            }
        }
        namespace util {
            class Lock;
        }
    }
    namespace __dbc {
        namespace memory {
            class Block;

            namespace protocol {
            }
        }
        namespace util {
            class Lock;
        }
    }

    namespace memory {
        using namespace gmac::__dbc::memory;
    }
    namespace util {
        using namespace gmac::__dbc::util;
    }
    
    using namespace __dbc;
}
#endif

#endif  /* GMAC_CONFIG_VISIBILITY_H */
