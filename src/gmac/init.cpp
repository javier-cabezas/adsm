#include "core/Process.h"
#include "util/Parameter.h"
#include "util/Private.h"
#include "util/Logger.h"
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

static void CONSTRUCTOR init(void)
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
#if defined(POSIX)
    osInit();
    threadInit();
    stdcInit();
#endif

    util::Logger::TRACE("Using %s memory manager", paramProtocol);
    util::Logger::TRACE("Using %s memory allocator", paramAllocator);
    // Process is a singleton class. The only allowed instance is Proc_
    util::Logger::TRACE("Initializing process");
    Process::create<Process>();
    apiInit();
    memoryInit(paramProtocol, paramAllocator);

    exitGmac();
}

static void DESTRUCTOR fini(void)
{
	gmac::enterGmac();
    gmac::util::Logger::TRACE("Cleaning GMAC");
    gmac::Process::destroy();
    delete _inGmacLock;
    // We do not exitGmac to allow proper stdc function handling
    gmac::util::Logger::Destroy();
}

} // namespace gmac

#if defined(_WIN32)
#include <windows.h>
 
// DLL entry function (called on load, unload, ...)
BOOL APIENTRY DllMain(HANDLE /*hModule*/, DWORD dwReason, LPVOID /*lpReserved*/)
{
	switch(dwReason) {
		case DLL_PROCESS_ATTACH:
			gmac::init();
			break;
		case DLL_PROCESS_DETACH:
			gmac::fini();
			break;
		case DLL_THREAD_ATTACH:
			// TODO: Handle thread creation -- Should be similar to pthread_create()
			break;
	};
    return TRUE;
}
#endif


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
