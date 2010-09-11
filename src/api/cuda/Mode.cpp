#include "Mode.h"
#include "Context.h"
#include "Accelerator.h"

namespace gmac { namespace cuda {

Mode::Mode(Accelerator *acc) :
    gmac::Mode(acc),
    acc(acc)
{
#ifdef USE_MULTI_CONTEXT
    __ctx = acc->createContext();
#endif

    switchIn();
    _context = new Context(acc, this);
    gmac::Mode::_context = _context;
#ifdef USE_MULTI_CONTEXT
    modules = ModuleDescriptor::createModules();
#else
    modules = acc->createModules();
#endif
    ModuleVector::const_iterator i;
    for(i = modules.begin(); i != modules.end(); i++) {
        (*i)->registerKernels(*this);
#ifdef USE_VM
        if((*i)->dirtyBitmap() != NULL) {
            _bitmapDevPtr = (*i)->dirtyBitmap()->devPtr();
            _bitmapShiftPageDevPtr = (*i)->dirtyBitmapShiftPage()->devPtr();
#ifdef BITMAP_BIT
            _bitmapShiftEntryDevPtr = (*i)->dirtyBitmapShiftEntry()->devPtr();
#endif
        }
#endif
    }
    switchOut();
}

Mode::~Mode()
{
    ModuleVector::const_iterator m;
    switchIn();
#ifdef USE_MULTI_CONTEXT
    for(m = modules.begin(); m != modules.end(); m++) {
        delete (*m);
    }
#endif
    delete _context;
    switchOut();
    modules.clear();
}

gmacError_t Mode::hostAlloc(void **addr, size_t size)
{
    switchIn();
    gmacError_t ret = acc->hostAlloc(addr, size);
    switchOut();
    return ret;
}

gmacError_t Mode::hostFree(void *addr)
{
    switchIn();
    gmacError_t ret = acc->hostFree(addr);
    switchOut();
    return ret;
}

void *Mode::hostMap(void *addr)
{
    switchIn();
    void *ret = acc->hostMap(addr);
    switchOut();
    return ret;
}

const Variable *Mode::constant(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
    for(m = modules.begin(); m != modules.end(); m++) {
        const Variable *var = (*m)->constant(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Variable *Mode::variable(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
    for(m = modules.begin(); m != modules.end(); m++) {
        const Variable *var = (*m)->variable(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Texture *Mode::texture(gmacTexture_t key) const
{
    ModuleVector::const_iterator m;
    for(m = modules.begin(); m != modules.end(); m++) {
        const Texture *tex = (*m)->texture(key);
        if(tex != NULL) return tex;
    }
    return NULL;
}

Stream Mode::eventStream() const {
    return _context->eventStream();
}

}}
