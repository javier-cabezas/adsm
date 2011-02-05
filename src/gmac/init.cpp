#include "core/Process.h"
#include "util/Parameter.h"
#include "util/Private.h"
#include "util/Logger.h"
#include "trace/Tracer.h"

#include "init.h"

namespace __impl {
util::Private<const char> _inGmac;
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

	util::Logger::Init();
    TRACE(GLOBAL, "Initialiazing GMAC");

    util::params::Init();
	gmac::trace::InitTracer();	
    trace::SetThreadState(trace::Running);

    /* Call initialization of interpose libraries */
#if defined(POSIX)
    osInit();
    threadInit();
#endif
    stdcInit();

#ifdef USE_MPI
    mpiInit();
#endif

    TRACE(GLOBAL, "Using %s memory manager", util::params::ParamProtocol);
    TRACE(GLOBAL, "Using %s memory allocator", util::params::ParamAllocator);
    // Process is a singleton class. The only allowed instance is Proc_
    TRACE(GLOBAL, "Initializing process");
    core::Process::create<__impl::core::Process>();
    core::apiInit();

    exitGmac();
}

static void DESTRUCTOR fini(void)
{
	gmac::enterGmac();
    TRACE(GLOBAL, "Cleaning GMAC");
    core::Process::destroy();
    gmac::trace::FiniTracer();
    delete _inGmacLock;
	// TODO: Clean-up logger
}

}

#if defined(_WIN32)
#include <windows.h>

static void InitThread()
{
	gmac::trace::StartThread("CPU");
	gmac::enterGmac();
	__impl::core::Process &proc = __impl::core::Process::getInstance();
	proc.initThread();
    gmac::trace::SetThreadState(__impl::trace::Running);
	gmac::exitGmac();
}

static void FiniThread()
{
	gmac::enterGmac();
	gmac::trace::SetThreadState(gmac::trace::Idle);	
	// Modes and Contexts already destroyed in Process destructor
	__impl::core::Process &proc = __impl::core::Process::getInstance();
	proc.finiThread();
	gmac::exitGmac();
}

// DLL entry function (called on load, unload, ...)
BOOL APIENTRY DllMain(HANDLE /*hModule*/, DWORD dwReason, LPVOID /*lpReserved*/)
{
	switch(dwReason) {
		case DLL_PROCESS_ATTACH:
            __impl::init();
			break;
		case DLL_PROCESS_DETACH:
			__impl::fini();
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
