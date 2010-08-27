#ifndef __KERNEL_MODE_IPP_
#define __KERNEL_MODE_IPP_

namespace gmac {

inline 
Mode *Mode::current()
{
    Mode *mode = static_cast<Mode *>(Mode::key.get());
    if(mode == NULL) mode = proc->create();
    gmac::util::Logger::ASSERTION(mode != NULL);
    return mode;
}


inline
void Mode::attach()
{
    Mode *mode = static_cast<Mode *>(Mode::key.get());
    if(mode == this) return;
    if(mode != NULL) mode->destroy();
    key.set(this);
}

inline
void Mode::detach()
{
    Mode *mode = static_cast<Mode *>(Mode::key.get());
    if(mode != NULL) mode->destroy();
    key.set(NULL);
}

inline
gmacError_t Mode::malloc(void **addr, size_t size, unsigned align)
{
    switchIn();
    gmacError_t ret = __acc->malloc(addr, size, align);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::free(void *addr)
{
    switchIn();
    gmacError_t ret = __acc->free(addr);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::copyToDevice(void *dev, const void *host, size_t size)
{
    switchIn();
    gmacError_t ret = __acc->copyToDevice(dev, host, size);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::copyToHost(void *host, const void *dev, size_t size)
{
    switchIn();
    gmacError_t ret = __acc->copyToHost(host, dev, size);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::copyDevice(void *dst, const void *src, size_t size)
{
    switchIn();
    gmacError_t ret = __acc->copyDevice(dst, src, size);
    switchOut();
    return ret;
}

}

#endif
