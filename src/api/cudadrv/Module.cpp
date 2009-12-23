#include "Module.h"

namespace gmac { namespace gpu {

const char *Module::pageTableSymbol = "__pageTable";

Function::Function(CUmodule mod, const char *name) :
    dev(dev),
    name(name)
{
    load(mod);
}

Variable::Variable(const char *dev, CUdeviceptr ptr, size_t size) :
    dev(dev),
    ptr(ptr),
    size(size)
{
}

Texture::Texture(CUmodule mod, struct __textureReference *ref, const char *name) :
    name(name),
    ref(ref)
{
    load(mod);
}

Module::Module(const void *fatBin) :
    fatBin(fatBin),
    _pageTable(NULL)
{
    TRACE("Module image: %p", fatBin);
    assert(cuModuleLoadFatBinary(&mod, fatBin) == CUDA_SUCCESS);
}

Module::Module(const Module &root) :
    fatBin(root.fatBin), functions(root.functions),
    variables(root.variables), constants(root.constants),
    textures(root.textures)
{
    reload();
}

Module::~Module() {
    assert(cuModuleUnload(mod) == CUDA_SUCCESS);
}

}}
