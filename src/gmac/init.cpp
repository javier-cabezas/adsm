#include "core/Process.h"
#include "util/Parameter.h"
#include "util/Private.h"
#include "util/Logger.h"
#include "trace/Tracer.h"

#include "init.h"

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

	gmac::util::Logger::Init();
    TRACE(GLOBAL, "Initialiazing GMAC");

    paramInit();
	gmac::trace::InitTracer();	

    /* Call initialization of interpose libraries */
#if defined(POSIX)
    osInit();
    threadInit();
#endif
    stdcInit();


    TRACE(GLOBAL, "Using %s memory manager", paramProtocol);
    TRACE(GLOBAL, "Using %s memory allocator", paramAllocator);
    // Process is a singleton class. The only allowed instance is Proc_
    TRACE(GLOBAL, "Initializing process");
    gmac::core::Process::create<gmac::core::Process>();
    gmac::core::apiInit();

    exitGmac();
}

static void DESTRUCTOR fini(void)
{
	gmac::enterGmac();
    TRACE(GLOBAL, "Cleaning GMAC");
    gmac::core::Process::destroy();
    delete _inGmacLock;
	// TODO: Clean-up logger
}

} // namespace gmac

#if defined(_WIN32)
#include <windows.h>

static void InitThread()
{
	gmac::trace::StartThread("CPU");
	gmac::enterGmac();
	gmac::core::Process &proc = gmac::core::Process::getInstance();
	proc.initThread();
	gmac::trace::SetThreadState(gmac::trace::Running);
	gmac::exitGmac();
}

static void FiniThread()
{
	gmac::enterGmac();
	gmac::trace::SetThreadState(gmac::trace::Idle);	
	// Modes and Contexts already destroyed in Process destructor
	gmac::core::Process &proc = gmac::core::Process::getInstance();
	proc.finiThread();
	gmac::exitGmac();
}

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
			InitThread();
			break;
		case DLL_THREAD_DETACH:			
			FiniThread();
			break;
	};
    return TRUE;
}


#endif


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
