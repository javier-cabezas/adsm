#ifndef __KERNEL_CONTEXT_IPP_
#define __KERNEL_CONTEXT_IPP_

inline void
Context::enable()
{
    PRIVATE_SET(key, this);
    _mm.realloc();
}

inline
Context *
Context::current()
{
    Context *ctx;
    ctx = static_cast<Context *>(PRIVATE_GET(key));
    if (ctx == NULL) ctx = Context::create();
    return ctx;
}

inline
bool
Context::hasCurrent()
{
    return PRIVATE_GET(key) != NULL;
}

inline gmacError_t
Context::error() const
{
    return _error;
}


inline unsigned
Context::id() const
{
    return _id;
}

inline void *
Context::bufferPageLocked() const
{
    return _bufferPageLocked;
}

inline size_t
Context::bufferPageLockedSize() const
{
    return _bufferPageLockedSize;
}

#endif
