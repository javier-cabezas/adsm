#include <csignal>

#include "../aspace.h"

namespace __impl { namespace hal { namespace cpu { namespace virt {

#if defined(LINUX)
static const int Signum_ = SIGSEGV;
#elif defined(DARWIN)
static const int Signum_ = SIGBUS;
#endif

// Change this to a per-address space basis
static struct sigaction defaultAction;
static hal_handler_sigsegv *handler;
static aspace *as;

void handler_sigsegv_main(int s, siginfo_t *info, void *ctx)
{
    handler->exec_pre();

    mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;
#if defined(LINUX)
    unsigned long isWrite = mCtx->gregs[REG_ERR] & 0x2;
#elif defined(DARWIN)
    unsigned long isWrite = (*mCtx)->__es.__err & 0x2;
#endif
    host_ptr addr = host_ptr(info->si_addr);

    if (!isWrite) TRACE(GLOBAL, "Read SIGSEGV for %p", addr);
    else TRACE(GLOBAL, "Write SIGSEGV for %p", addr);

    bool resolved = false;

    auto it = as->get_map_addr_to_view().upper_bound(hal::ptr::offset_type(addr));

    if (it != as->get_map_addr_to_view().end() &&
        it->second->get_offset() <= hal::ptr::offset_type(addr) &&
        (it->second->get_offset() + it->second->get_object().get_size() > hal::ptr::offset_type(addr))) {

        hal::ptr p(*it->second, hal::ptr::offset_type(addr) - it->second->get_offset());
        resolved = handler->exec(p, isWrite);
    }

    if (resolved == false) {
        fprintf(stderr, "Uoops! I could not find a mapping for %p. I will abort the execution\n", addr);
        abort();
        // TODO: set the signal mask and other stuff
        if (defaultAction.sa_flags & SA_SIGINFO)
            return defaultAction.sa_sigaction(s, info, ctx);
        return defaultAction.sa_handler(s);
    }

    handler->exec_post();
}

void
aspace::handler_sigsegv_overload()
{
    struct sigaction segvAction;
    ::memset(&segvAction, 0, sizeof(segvAction));
    segvAction.sa_sigaction = handler_sigsegv_main;
    segvAction.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&segvAction.sa_mask);

    if (sigaction(Signum_, &segvAction, &defaultAction) < 0)
        FATAL("sigaction: %s", strerror(errno));

    as = this;
    handler = &as->handlers_.back();

    TRACE(LOCAL, "New signal handler programmed");
}

void
aspace::handler_sigsegv_restore()
{
    if (sigaction(Signum_, &defaultAction, NULL) < 0)
        FATAL("sigaction: %s", strerror(errno));

    as = nullptr;

    TRACE(LOCAL, "Old signal handler restored");
}
 
}}}}
