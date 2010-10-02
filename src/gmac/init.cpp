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

#include "util/Parameter.h"
#include "util/Private.h"
#include "util/Logger.h"
#include "util/FileLock.h"

#include "trace/Function.h"

#include "init.h"

#ifdef PARAVER
namespace paraver {
extern int init;
}
#endif

namespace gmac {
gmac::util::Private<const char> _inGmac;
GMACLock * _inGmacLock;

const char _gmacCode = 1;
const char _userCode = 0;

char _gmacInit = 0;

#ifdef LINUX 
#define GLOBAL_FILE_LOCK "/tmp/gmacSystemLock"
#else
#ifdef DARWIN
#define GLOBAL_FILE_LOCK "/tmp/gmacSystemLock"
#endif
#endif

static void __attribute__((constructor))
init(void)
{
	util::Private<const char>::init(_inGmac);
    _inGmacLock = new GMACLock();

	enterGmac();
    _gmacInit = 1;

    util::Logger::Create("GMAC");
    util::Logger::TRACE("Initialiazing GMAC");

#ifdef PARAVER
    paraver::init = 1;
#endif
    //util::FileLock(GLOBAL_FILE_LOCK, trace::LockSystem);

    //FILE * lockSystem;

    paramInit();
    trace::Function::init();

    /* Call initialization of interpose libraries */
    osInit();
    threadInit();
    stdcInit();

    util::Logger::TRACE("Using %s memory manager", paramProtocol);
    util::Logger::TRACE("Using %s memory allocator", paramAllocator);
    Process::init(paramProtocol, paramAllocator);
    exitGmac();
}

static void __attribute__((destructor))
fini(void)
{
	gmac::enterGmac();
    gmac::util::Logger::TRACE("Cleaning GMAC");
    gmac::Process::fini();
    delete _inGmacLock;
    // We do not exitGmac to allow proper stdc function handling
    gmac::util::Logger::Destroy();
}

} // namespace gmac



/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
