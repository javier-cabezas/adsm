#include "core/hpe/Process.h"

#include "memory/Allocator.h"
#include "memory/Handler.h"
#include "memory/Manager.h"
#include "memory/allocator/Slab.h"

#include "util/Parameter.h"
#include "util/Private.h"
#include "util/Logger.h"

#include "trace/Tracer.h"

#include "init.h"

using __impl::memory::allocator::Slab;

class GMAC_LOCAL GMACLock : public gmac::util::RWLock {
public:
    GMACLock() : gmac::util::RWLock("Process") {}

    void lockRead()  const { gmac::util::RWLock::lockRead();  }
    void lockWrite() const { gmac::util::RWLock::lockWrite(); }
    void unlock()    const { gmac::util::RWLock::unlock();   }
};

static __impl::util::Private<const char> inGmac_;
static GMACLock * inGmacLock;

static const char gmacCode = 1;
static const char userCode = 0;

static Atomic gmacInit__ = 0;
static Atomic gmacFini__ = -1;

#ifdef LINUX 
#define GLOBAL_FILE_LOCK "/tmp/gmacSystemLock"
#else
#ifdef DARWIN
#define GLOBAL_FILE_LOCK "/tmp/gmacSystemLock"
#endif
#endif

static Process *Process_ = NULL;
static Manager *Manager_ = NULL;
static Allocator *Allocator_ = NULL;

extern void CUDA(Process &);
extern void OpenCL(Process &);

static void init(void)
{
    /* Create GMAC enter lock and set GMAC as initialized */
    inGmacLock = new GMACLock();
    __impl::util::Private<const char>::init(inGmac_);
    enterGmac();

    /* Call initialization of interpose libraries */
#if defined(POSIX)
    osInit();
    threadInit();
#endif
    stdcInit();

#ifdef USE_MPI
    mpiInit();
#endif

    TRACE(GLOBAL, "Using %s memory manager", __impl::util::params::ParamProtocol);
    TRACE(GLOBAL, "Using %s memory allocator", __impl::util::params::ParamAllocator);

    // Set the entry and exit points for Manager
    __impl::memory::Handler::setEntry(enterGmac);
    __impl::memory::Handler::setExit(exitGmac);

    // Process is a singleton class. The only allowed instance is Proc_
    TRACE(GLOBAL, "Initializing process");
    Process_ = new Process();

    TRACE(GLOBAL, "Initializing memory");
    Manager_ = new Manager(*Process_);
    Allocator_ = new Slab(*Manager_);

#if defined(USE_CUDA)
    TRACE(GLOBAL, "Initializing CUDA");
    CUDA(*Process_);
#endif
#if defined(USE_OPENCL)
    TRACE(GLOBAL, "Initializing OpenCL");
    OpenCL(*Process_);
#endif


    exitGmac();
}

void enterGmac()
{
	if(AtomicTestAndSet(gmacInit__, 0, 1) == 0) init();
    inGmac_.set(&gmacCode);
    inGmacLock->lockRead();
}


void enterGmacExclusive()
{
    inGmac_.set(&gmacCode);
    inGmacLock->lockWrite();
}

void exitGmac()
{
    inGmacLock->unlock();
    inGmac_.set(&userCode);
}

char inGmac()
{ 
    if(gmacInit__ == 0) return 1;
    char *ret = (char  *)inGmac_.get();
    if(ret == NULL) return 0;
    else if(*ret == gmacCode) return 1;
    return 0;
}

Process &getProcess()
{
    return *Process_;
}

Manager &getManager()
{
    return *Manager_;
}

Allocator &getAllocator()
{
    return *Allocator_;
}

__impl::core::hpe::Mode &getCurrentMode()
{
    return Process_->getCurrentMode();
}

DESTRUCTOR(fini);
static void fini(void)
{
	enterGmac();
    if(AtomicInc(gmacFini__) == 0) {
        Allocator_->destroy();
        Manager_->destroy();
        Process_->destroy();
        delete inGmacLock;
    }
	// TODO: Clean-up logger
}


#if defined(_WIN32)
#include <windows.h>

static void InitThread()
{
	gmac::trace::StartThread("CPU");
	enterGmac();
	getProcess().initThread();
    gmac::trace::SetThreadState(__impl::trace::Running);
	exitGmac();
}

static void FiniThread()
{
	enterGmac();
	gmac::trace::SetThreadState(gmac::trace::Idle);	
	// Modes and Contexts already destroyed in Process destructor
	getProcess().finiThread();
	exitGmac();
}

// DLL entry function (called on load, unload, ...)
BOOL APIENTRY DllMain(HANDLE /*hModule*/, DWORD dwReason, LPVOID /*lpReserved*/)
{
	switch(dwReason) {
		case DLL_PROCESS_ATTACH:
            break;
		case DLL_PROCESS_DETACH:
			// Really ugly hack -- Stupid windows do not allow calling DLLs from static
			// destructors, so we cannot release resources at termination time
			AtomicInc(gmacFini__);
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


namespace __impl {


}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
