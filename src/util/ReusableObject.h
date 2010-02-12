/*
 * Copyright (C) 2009 {{{
 *      - Javier Cabezas <jcabezas@ac.upc.edu>
 *      - Isaac Gelado <igelado@ac.upc.edu>
 *      - Lluís Vilanova <vilanova@ac.upc.edu>
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
 *
 * This file is part of OSS/1 (Operating System Simulator 1)
 */

#ifndef CYCLE_UTILS_REUSABLE_OBJECT_H
#define CYCLE_UTILS_REUSABLE_OBJECT_H

#include <cstddef>

namespace gmac {
    namespace util {

        /*! \todo Delete ? */
        template <typename Type>
        class Pool {
            template <typename Type2> friend class ReusableObject;
        public:
            Pool()
                : freeList(NULL)
                {}

            ~Pool()
                {
                    Object * next, * tmp;
                    for (next = freeList; next; next = tmp) {
                        tmp = next->next;
                        delete next;
                    }
                }

            union Object {
                char shit[sizeof(Type)];
                Object * next;
            };

        private:
            Object * freeList;

        };

        /*! \todo Delete ? */
        template <typename Type>
        class ReusableObject {
        public:
            void * operator new (size_t bytes)
                {
                    typename Pool<Type>::Object * res = pool.freeList;
                    return res? (pool.freeList = pool.freeList->next, res) : new typename Pool<Type>::Object;
                }

            void operator delete (void *ptr)
                {
                    ((typename Pool<Type>::Object *) ptr)->next = pool.freeList;
                    pool.freeList = (typename Pool<Type>::Object *) ptr;
                }
  
        protected:
            static Pool<Type> pool;
        };

// Static initialization

        template<class T> Pool<T> 
        ReusableObject<T>::pool;

    };
};

#endif /* CYCLE_UTILS_REUSABLE_OBJECT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=100 foldmethod=marker expandtab cindent cinoptions=p5,t0,(0: */
