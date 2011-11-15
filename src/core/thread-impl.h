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

#ifndef GMAC_CORE_THREAD_IMPL_H_
#define GMAC_CORE_THREAD_IMPL_H_

#include "util/Logger.h"
#include "util/Private.h"

namespace __impl { namespace core {

inline void
TLS::Init()
{
    __impl::util::Private<thread>::init(CurrentThread_);
}

inline
bool
TLS::has_current_thread()
{
    return TLS::CurrentThread_.get() != NULL;
}

inline
thread &
TLS::get_current_thread()
{
    ASSERTION(TLS::CurrentThread_.get() != NULL);
    return *TLS::CurrentThread_.get();
}

inline
thread::thread() :
    lastError_(gmacSuccess)
{
    ASSERTION(TLS::CurrentThread_.get() == NULL);
    TLS::CurrentThread_.set(this);
#ifdef DEBUG
    debugTID_ = THREAD_T(AtomicInc(thread::NextTID_));
#endif
}

inline
thread::~thread()
{
    TLS::CurrentThread_.set(NULL);
}

#ifdef DEBUG
inline
THREAD_T
thread::get_debug_tid()
{
    if (TLS::CurrentThread_.get() == NULL) return THREAD_T(1);
    return TLS::get_current_thread().debugTID_;
}
#endif


inline
gmacError_t &
thread::get_last_error()
{
    return TLS::get_current_thread().lastError_;
}

inline
void
thread::set_last_error(gmacError_t error)
{
    TLS::get_current_thread().lastError_ = error;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
