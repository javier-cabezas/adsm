/*
 * <+ DESCRIPTION +>
 *
 * Copyright (C) 2012, Javier Cabezas <jcabezas in ac upc edu> {{{
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

#ifndef GMAC_HAL_COPY_IMPL_H
#define GMAC_HAL_COPY_IMPL_H

#include "types.h"

namespace __impl { namespace hal {

event_ptr 
copy(ptr dst, const_ptr src, size_t count, stream &stream, list_event &_dependencies, gmacError_t &err)
{
    event_ptr ret = dst.get_view().get_vaspace().copy(dst, src, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

event_ptr 
copy(ptr dst, const_ptr src, size_t count, stream &stream, event_ptr event, gmacError_t &err)
{
    list_event list;
    list.add_event(event);

    event_ptr ret = dst.get_view().get_vaspace().copy(dst, src, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

event_ptr 
copy(ptr dst, const_ptr src, size_t count, stream &stream, gmacError_t &err)
{
    return dst.get_view().get_vaspace().copy(dst, src, count, stream, NULL, err);
}

event_ptr
copy(ptr dst, device_input &input, size_t count, stream &stream, list_event &_dependencies, gmacError_t &err)
{
    event_ptr ret = dst.get_view().get_vaspace().copy(dst, input, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

event_ptr
copy(ptr dst, device_input &input, size_t count, stream &stream, event_ptr event, gmacError_t &err)
{
    list_event list;
    list.add_event(event);

    event_ptr ret = dst.get_view().get_vaspace().copy(dst, input, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

event_ptr
copy(ptr dst, device_input &input, size_t count, stream &stream, gmacError_t &err)
{
    return dst.get_view().get_vaspace().copy(dst, input, count, stream, NULL, err);
}

event_ptr
copy(device_output &output, const_ptr src, size_t count, stream &stream, list_event &_dependencies, gmacError_t &err)
{
    event_ptr ret = src.get_view().get_vaspace().copy(output, src, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

event_ptr
copy(device_output &output, const_ptr src, size_t count, stream &stream, event_ptr event, gmacError_t &err)
{
    list_event list;
    list.add_event(event);

    event_ptr ret = src.get_view().get_vaspace().copy(output, src, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

event_ptr
copy(device_output &output, const_ptr src, size_t count, stream &stream, gmacError_t &err)
{
    return src.get_view().get_vaspace().copy(output, src, count, stream, NULL, err);
}

event_ptr 
copy_async(ptr dst, const_ptr src, size_t count, stream &stream, list_event &_dependencies, gmacError_t &err)
{
    event_ptr ret = dst.get_view().get_vaspace().copy_async(dst, src, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

event_ptr 
copy_async(ptr dst, const_ptr src, size_t count, stream &stream, event_ptr event, gmacError_t &err)
{
    list_event list;
    list.add_event(event);

    event_ptr ret = dst.get_view().get_vaspace().copy_async(dst, src, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

event_ptr 
copy_async(ptr dst, const_ptr src, size_t count, stream &stream, gmacError_t &err)
{
    return dst.get_view().get_vaspace().copy_async(dst, src, count, stream, NULL, err);
}

event_ptr
copy_async(ptr dst, device_input &input, size_t count, stream &stream, list_event &_dependencies, gmacError_t &err)
{
    event_ptr ret = dst.get_view().get_vaspace().copy_async(dst, input, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

event_ptr
copy_async(ptr dst, device_input &input, size_t count, stream &stream, event_ptr event, gmacError_t &err)
{
    list_event list;
    list.add_event(event);

    event_ptr ret = dst.get_view().get_vaspace().copy_async(dst, input, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

event_ptr
copy_async(ptr dst, device_input &input, size_t count, stream &stream, gmacError_t &err)
{
    return dst.get_view().get_vaspace().copy_async(dst, input, count, stream, NULL, err);
}

event_ptr
copy_async(device_output &output, const_ptr src, size_t count, stream &stream, list_event &_dependencies, gmacError_t &err)
{
    event_ptr ret = src.get_view().get_vaspace().copy_async(output, src, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

event_ptr
copy_async(device_output &output, const_ptr src, size_t count, stream &stream, event_ptr event, gmacError_t &err)
{
    list_event list;
    list.add_event(event);

    event_ptr ret = src.get_view().get_vaspace().copy_async(output, src, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

event_ptr
copy_async(device_output &output, const_ptr src, size_t count, stream &stream, gmacError_t &err)
{
    return src.get_view().get_vaspace().copy_async(output, src, count, stream, NULL, err);
}

event_ptr 
memset(ptr dst, int c, size_t count, stream &stream, list_event &_dependencies, gmacError_t &err)
{
    event_ptr ret = dst.get_view().get_vaspace().memset(dst, c, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

event_ptr 
memset(ptr dst, int c, size_t count, stream &stream, event_ptr event, gmacError_t &err)
{
    list_event list;
    list.add_event(event);

    event_ptr ret = dst.get_view().get_vaspace().memset(dst, c, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

event_ptr 
memset(ptr dst, int c, size_t count, stream &stream, gmacError_t &err)
{
    return dst.get_view().get_vaspace().memset(dst, c, count, stream, NULL, err);
}

event_ptr 
memset_async(ptr dst, int c, size_t count, stream &stream, list_event &_dependencies, gmacError_t &err)
{
    event_ptr ret = dst.get_view().get_vaspace().memset_async(dst, c, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

event_ptr 
memset_async(ptr dst, int c, size_t count, stream &stream, event_ptr event, gmacError_t &err)
{
    list_event list;
    list.add_event(event);

    event_ptr ret = dst.get_view().get_vaspace().memset_async(dst, c, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

event_ptr 
memset_async(ptr dst, int c, size_t count, stream &stream, gmacError_t &err)
{
    return dst.get_view().get_vaspace().memset_async(dst, c, count, stream, NULL, err);
}

}}

#endif /* COPY_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
