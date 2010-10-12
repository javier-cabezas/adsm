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

#include "api/cuda/Accelerator.h"
#include "api/cuda/Mode.h"

#include "memory/Bitmap.h"

#ifdef USE_VM
namespace gmac { namespace memory { namespace vm {

void Bitmap::allocate()
{
    assertion(device_ == NULL);
    gmac::cuda::Mode * mode = gmac::cuda::Mode::current();
#ifdef USE_HOSTMAP_VM
    mode->hostAlloc((void **)&_bitmap, size_);
    device_ = mode->hostMap(_bitmap);
    memset(_bitmap, 0, size());
    trace("Allocating dirty bitmap (%zu bytes)", size());
#else
    mode->malloc((void **)&device_, size_);
    trace("Allocating dirty bitmap %p -> %p (%zu bytes)", _bitmap, device_, size_);
#endif
}

Bitmap::~Bitmap()
{
    gmac::cuda::Mode * mode = gmac::cuda::Mode::current();
    if(device_ != NULL) mode->hostFree(_bitmap);
}

void
Bitmap::syncHost()
{
#ifndef USE_HOSTMAP_VM
    trace("Syncing Bitmap");
    Mode * mode = Mode::current();

    gmac::memory::vm::Bitmap & bitmap = mode->dirtyBitmap();
    trace("Setting dirty bitmap on host: %p -> %p: %zd", (void *) cuda::Accelerator::gpuAddr(bitmap.device()), bitmap.host(), bitmap.size());
    gmacError_t ret;
    //printf("Bitmap toHost\n");
    ret = mode->copyToHost(bitmap.host(), bitmap.device(), bitmap.size());
    cfatal(ret == gmacSuccess, "Unable to copy back dirty bitmap");
    bitmap.reset();
#endif
}

void
Bitmap::syncDevice()
{
#ifndef USE_HOSTMAP_VM
    trace("Syncing Bitmap");
    gmac::cuda::Mode * mode = gmac::cuda::Mode::current();

    gmac::memory::vm::Bitmap & bitmap = mode->dirtyBitmap();
    trace("Setting dirty bitmap on device: %p -> %p (0x%lx): %zd", (void *) cuda::Accelerator::gpuAddr(bitmap.device()), bitmap.host(), mode->dirtyBitmapDevPtr(), bitmap.size());
    gmacError_t ret;
    ret = mode->copyToDevice((void *) mode->dirtyBitmapDevPtr(), &device_, sizeof(void *));
    cfatal(ret == gmacSuccess, "Unable to set the pointer in the device %p", (void *) mode->dirtyBitmapDevPtr());
    ret = mode->copyToDevice((void *) mode->dirtyBitmapShiftPageDevPtr(), &_shiftPage, sizeof(int));
    cfatal(ret == gmacSuccess, "Unable to set shift page in the device %p", (void *) mode->dirtyBitmapShiftPageDevPtr());
#ifdef BITMAP_BIT
    ret = mode->copyToDevice((void *) mode->dirtyBitmapShiftEntryDevPtr(), &_shiftEntry, sizeof(int));
    cfatal(ret == gmacSuccess, "Unable to set shift entry in the device %p", (void *) mode->dirtyBitmapShiftEntryDevPtr());
#endif

    //printf("Bitmap toHost\n");
    ret = mode->copyToDevice(bitmap.device(), bitmap.host(), bitmap.size());
    cfatal(ret == gmacSuccess, "Unable to copy dirty bitmap to device");

    _synced = false;
#endif
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
