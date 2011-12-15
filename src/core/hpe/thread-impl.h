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

#ifndef GMAC_CORE_THREAD_HPE_IMPL_H_
#define GMAC_CORE_THREAD_HPE_IMPL_H_

#include "hpe/init.h"

#include "core/hpe/process.h"

#include "util/private.h"

#include "vdevice.h"

namespace __impl { namespace core { namespace hpe {

inline
thread &
thread::get_current_thread()
{
    if (has_current_thread() == false) {
        getProcess().initThread(true, 0);
    }
    return static_cast<thread &>(TLS::get_current_thread());
}

inline
bool
thread::has_current_thread()
{
    return TLS::has_current_thread();
}

#if 0
inline
vdevice_table &
thread::getCurrentVirtualDeviceTable()
{
    return get_current_thread().vDeviceTable_;
}
#endif

inline
thread::thread(process &proc) :
    core::thread(),
    process_(proc),
    currentVirtualDevice_(NULL)
{
}

inline
vdevice *
thread::get_virtual_device(GmacVirtualDeviceId id)
{
    vdevice *ret = NULL;

    map_vdevice::iterator it;
    it = mapVDevices_.find(id);
    if (it != mapVDevices_.end()) {
        ret = it->second;
    }

    return ret;
}

inline
vdevice &
thread::get_current_virtual_device() const
{
    ASSERTION(currentVirtualDevice_ != NULL);
    vdevice *ret = currentVirtualDevice_;
    return *ret;
}

inline
gmacError_t
thread::add_virtual_device(GmacVirtualDeviceId id, vdevice &dev)
{
    gmacError_t ret;

    ret = gmacSuccess;

    map_vdevice::iterator it;
    it = mapVDevices_.find(id);
    if (it == mapVDevices_.end()) {
        mapVDevices_.insert(map_vdevice::value_type(id, &dev));
    } else {
        ret = gmacErrorInvalidValue;
    }

    return ret;
}

inline
gmacError_t
thread::remove_virtual_device(vdevice &dev)
{
    ASSERTION(currentVirtualDevice_ != NULL);

    gmacError_t ret;

    ret = gmacSuccess;
    map_vdevice::iterator it;
    it = mapVDevices_.find(dev.get_id());
    if (it != mapVDevices_.end()) {
        mapVDevices_.erase(it);
    } else {
        ret = gmacErrorInvalidValue;
    }

    return ret;
}

inline
void
thread::set_current_virtual_device(vdevice &dev)
{
    currentVirtualDevice_ = &dev;
}

#if 0
inline
context *
thread::get_context(address_space &aspace)
{
    context *ret = NULL;
    map_context &mapContexts = mapContexts_;

    map_context::iterator it;
    it = mapContexts.find(&aspace);
    if (it != mapContexts.end()) {
        ret = it->second;
    }

    return ret;
}
#endif

inline gmacError_t
thread::new_kernel_launch(hal::kernel_t::config *config, kernel::arg_list *args)
{
    return new_kernel_launch(config, args, get_current_virtual_device());
}

inline gmacError_t
thread::new_kernel_launch(hal::kernel_t::config *config, kernel::arg_list *args, vdevice &dev)
{
    map_config::iterator it = mapConfigs_.find(&dev);

    if (it == mapConfigs_.end()) {
        mapConfigs_.insert(map_config::value_type(&dev, pair_launch(config, args)));
    } else {
        it->second.first  = config;
        it->second.second = args;
    }

    return gmacSuccess;
}

inline thread::pair_launch &
thread::get_kernel_launch()
{
    return get_kernel_launch(get_current_virtual_device());
}

inline thread::pair_launch &
thread::get_kernel_launch(vdevice &dev)
{
    map_config::iterator it = mapConfigs_.find(&dev);

    ASSERTION(it != mapConfigs_.end());

    return it->second;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
