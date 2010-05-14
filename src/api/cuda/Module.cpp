#include "Module.h"

#include <gmac/init.h>

namespace gmac { namespace gpu {

ModuleDescriptor::ModuleDescriptorVector ModuleDescriptor::Modules;

#ifdef USE_VM
const char *Module::dirtyBitmapSymbol = "__dirtyBitmap";
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
    logger.assertion(ret == CUDA_SUCCESS);
    _size = tmp;
}

Texture::Texture(const TextureDescriptor & t, CUmodule mod) :
    TextureDescriptor(t.name(), t.key())
{
    CUresult ret = cuModuleGetTexRef(&_texRef, mod, name());
    logger.assertion(ret == CUDA_SUCCESS);
}

ModuleDescriptor::ModuleDescriptor(const void *fatBin) :
    logger("ModuleDescriptor"),
    _fatBin(fatBin)
{
    logger.trace("Creating module descriptor: %p", _fatBin);
    Modules.push_back(this);
}

ModuleVector
ModuleDescriptor::createModules()
{
    ::logger->trace("Creating modules");
    ModuleVector modules;

    ModuleDescriptorVector::const_iterator it;
    for (it = Modules.begin(); it != Modules.end(); it++) {
        ::logger->trace("Creating module: %p", (*it)->_fatBin);
        modules.push_back(new Module(*(*it)));
    }
    return modules;
}

Module::Module(const ModuleDescriptor & d) :
    logger("Module"),
    _fatBin(d._fatBin)
{
    logger.trace("Module image: %p", _fatBin);
    CUresult res;
    res = cuModuleLoadFatBinary(&_mod, _fatBin);
    logger.cfatal(res == CUDA_SUCCESS, "Error loading module: %d", res);

    Context * ctx = Context::current();
    ModuleDescriptor::KernelVector::const_iterator k;
    for (k = d._kernels.begin(); k != d._kernels.end(); k++) {
        Kernel * kernel = new Kernel(*k, _mod);
        ctx->kernel(k->key(), kernel);
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
            logger.trace("Found constant to set a dirty bitmap on device");
        }
#endif
    }

    ModuleDescriptor::TextureVector::const_iterator t;
    for (t = d._textures.begin(); t != d._textures.end(); t++) {
        _textures.insert(TextureMap::value_type(t->key(), Texture(*t, _mod)));
    }

}

Module::~Module() {
    CUresult ret = cuModuleUnload(_mod);
    logger.assertion(ret == CUDA_SUCCESS);
}

}}
