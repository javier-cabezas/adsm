#include "Module.h"
#include "Mode.h"

#include <gmac/init.h>

namespace gmac { namespace gpu {

ModuleDescriptor::ModuleDescriptorVector ModuleDescriptor::Modules;

#ifdef USE_VM
const char *Module::dirtyBitmapSymbol = "__dirtyBitmap";
const char *Module::shiftPageSymbol  = "__SHIFT_PAGE";
const char *Module::shiftEntrySymbol = "__SHIFT_ENTRY";
#endif

VariableDescriptor::VariableDescriptor(const char *name, gmacVariable_t key, bool constant) :
    Descriptor<gmacVariable_t>(name, key),
    _constant(constant)
{
}

Variable::Variable(const VariableDescriptor & v, CUmodule mod) :
    VariableDescriptor(v.name(), v.key(), v.constant())
{
    unsigned int tmp;
    CUresult ret = cuModuleGetGlobal(&_ptr, &tmp, mod, name());
    assertion(ret == CUDA_SUCCESS);
    _size = tmp;
}

Texture::Texture(const TextureDescriptor & t, CUmodule mod) :
    TextureDescriptor(t.name(), t.key())
{
    CUresult ret = cuModuleGetTexRef(&_texRef, mod, name());
    assertion(ret == CUDA_SUCCESS);
}

ModuleDescriptor::ModuleDescriptor(const void *fatBin) :
    _fatBin(fatBin)
{
    trace("Creating module descriptor: %p", _fatBin);
    Modules.push_back(this);
}

ModuleDescriptor::~ModuleDescriptor()
{
    _kernels.clear();
    _variables.clear();
    _constants.clear();
    _textures.clear();
}

ModuleVector
ModuleDescriptor::createModules()
{
    util::Logger::TRACE("Creating modules");
    ModuleVector modules;

    ModuleDescriptorVector::const_iterator it;
    for (it = Modules.begin(); it != Modules.end(); it++) {
        util::Logger::TRACE("Creating module: %p", (*it)->_fatBin);
        modules.push_back(new Module(*(*it)));
    }
    return modules;
}

Module::Module(const ModuleDescriptor & d) :
    _fatBin(d._fatBin)
{
    trace("Module image: %p", _fatBin);
    CUresult res;
    res = cuModuleLoadFatBinary(&_mod, _fatBin);
    cfatal(res == CUDA_SUCCESS, "Error loading module: %d", res);

    ModuleDescriptor::KernelVector::const_iterator k;
    for (k = d._kernels.begin(); k != d._kernels.end(); k++) {
        Kernel * kernel = new Kernel(*k, _mod);
        _kernels.insert(KernelMap::value_type(k->key(), kernel));
    }

    ModuleDescriptor::VariableVector::const_iterator v;
    for (v = d._variables.begin(); v != d._variables.end(); v++) {
        _variables.insert(VariableMap::value_type(v->key(), Variable(*v, _mod)));
    }

    for (v = d._constants.begin(); v != d._constants.end(); v++) {
        _constants.insert(VariableMap::value_type(v->key(), Variable(*v, _mod)));
#ifdef USE_VM
        if(strncmp(v->name(), dirtyBitmapSymbol, strlen(dirtyBitmapSymbol)) == 0) {
            __dirtyBitmap = &_constants.find(v->key())->second;
            trace("Found constant to set a dirty bitmap on device");
        }

        if(strncmp(v->name(), shiftPageSymbol, strlen(shiftPageSymbol)) == 0) {
            __shiftPage = &_constants.find(v->key())->second;
            trace("Found constant to set __SHIFT_PAGE");

            size_t tmp = ctx.mm().dirtyBitmap().shiftPage();
            res = cuMemcpyHtoD(__shiftPage->devPtr(), &tmp, sizeof(size_t));
            cfatal(res == CUDA_SUCCESS, "Unable to set shift page");
        }

        if(strncmp(v->name(), shiftEntrySymbol, strlen(shiftEntrySymbol)) == 0) {
            __shiftEntry = &_constants.find(v->key())->second;
            trace("Found constant to set __SHIFT_ENTRY");

            size_t tmp = ctx.mm().dirtyBitmap().shiftEntry();
            res = cuMemcpyHtoD(__shiftEntry->devPtr(), &tmp, sizeof(size_t));
            cfatal(res == CUDA_SUCCESS, "Unable to set shift entry");
        }
#endif
    }

    ModuleDescriptor::TextureVector::const_iterator t;
    for (t = d._textures.begin(); t != d._textures.end(); t++) {
        _textures.insert(TextureMap::value_type(t->key(), Texture(*t, _mod)));
    }

}

Module::~Module()
{
    CUresult ret = cuModuleUnload(_mod);
    assertion(ret == CUDA_SUCCESS);
    _variables.clear();
    _constants.clear();
    _textures.clear();

    KernelMap::iterator i;
    for(i = _kernels.begin(); i != _kernels.end(); i++) delete i->second;
    _kernels.clear();
}

void Module::registerKernels(Mode &mode) const
{
    KernelMap::const_iterator k;
    for (k = _kernels.begin(); k != _kernels.end(); k++) {
        mode.kernel(k->first, k->second);
    }
}

}}
