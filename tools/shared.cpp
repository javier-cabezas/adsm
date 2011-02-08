/*
 * <+ DESCRIPTION +>
 *
 * Copyright (C) 2011, Javier Cabezas <jcabezas in ac upc edu> {{{
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

#include <cstdio>

extern "C" {
    extern char __gmac_ocl_code_start __attribute__((weak));
    extern char __gmac_ocl_code_end   __attribute__((weak));
};

int print()
{
    size_t size = &__gmac_ocl_code_end - &__gmac_ocl_code_start;
    char *start = &__gmac_ocl_code_start;

    printf("%p-%p\n", &__gmac_ocl_code_start, &__gmac_ocl_code_end);
    printf("%zd\n", size);
    printf("%s\n", start);

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
