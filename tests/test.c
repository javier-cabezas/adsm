/*
 * <+ DESCRIPTION +>
 *
 * Copyright (C) 2010, Javier Cabezas <jcabezas in ac upc edu> {{{
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

#include <signal.h>
#include <stdlib.h>

#include "test.h"

#define __ctor__ __attribute__((constructor))
#define __dtor__ __attribute__((destructor))

char * result_str[] = {
    "PASSED",
    "FAILED",
};

char * result_str_unknown = "UNKNOWN";

void check_result(int status, void * null)
{
    if (status != 0) status = 1;
    printf("%s\n", result_str[status]);
}

struct sigaction default_sigabrt;

void sig_abort(int s, siginfo_t *info, void * null)
{
    check_result(GMAC_TEST_FAILED, NULL);
}

void __ctor__ register_on_exit()
{
    struct sigaction my_sigabrt;
    memset(&my_sigabrt, 0, sizeof(my_sigabrt));
    my_sigabrt.sa_sigaction = sig_abort;
    my_sigabrt.sa_flags     = SA_SIGINFO | SA_RESTART;

	if(sigaction(SIGABRT, &my_sigabrt, &default_sigabrt) < 0)
        abort();

    on_exit(check_result, NULL);
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
