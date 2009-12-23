#ifndef __GPU_MODULE_IPP_
#define __GPU_MODULE_IPP_

inline void
Function::load(CUmodule mod)
{
    assert(cuModuleGetFunction(&fun, mod, name) == CUDA_SUCCESS);
}

inline void
Texture::load(CUmodule mod)
{
    assert(cuModuleGetTexRef(&ref->__texref, mod, name) == CUDA_SUCCESS);
}

inline void
Module::reload()
{
    TRACE("Module image: %p", fatBin);
    CUresult r = cuModuleLoadFatBinary(&mod, fatBin);
    assert(r == CUDA_SUCCESS);
    FunctionMap::iterator f;
    for(f = functions.begin(); f != functions.end(); f++)
        f->second.load(mod);
    TextureList::iterator t;
    for(t = textures.begin(); t != textures.end(); t++)
        t->load(mod);
}

inline void
Module::function(const char *host, const char *dev)
{
    functions.insert(FunctionMap::value_type(host, Function(mod, dev)));
}

inline const Function *
Module::function(const char *name) const
{
    FunctionMap::const_iterator f;
    f = functions.find(name);
    if(f == functions.end()) return NULL;
    return &f->second;
}

inline void
Module::variable(const char *host, const char *dev)
{
    CUdeviceptr ptr; 
    unsigned int size;
    CUresult ret = cuModuleGetGlobal(&ptr, &size, mod, dev);
    variables.insert(VariableMap::value_type(host, Variable(dev, ptr, size)));
}

inline const Variable *
Module::variable(const char *name) const
{
    VariableMap::const_iterator v;
    v = variables.find(name);
    if(v == variables.end()) return NULL;
    return &v->second;
}

inline void
Module::constant(const char *host, const char *dev)
{
	CUdeviceptr ptr; 
	unsigned int size;
	std::pair<VariableMap::iterator, bool> var;
	CUresult ret = cuModuleGetGlobal(&ptr, &size, mod, dev);
	var = constants.insert(VariableMap::value_type(host,
			Variable(dev, ptr, size)));
    if(strncmp(dev, pageTableSymbol, strlen(pageTableSymbol)) == 0)
        _pageTable = &var.first->second;
}

inline const Variable *
Module::constant(const char *name) const
{
    VariableMap::const_iterator v;
    v = constants.find(name);
    if(v == constants.end()) return NULL;
    return &v->second;
}

inline Variable *
Module::pageTable() const 
{
    return _pageTable;
}

inline void
Module::texture(struct __textureReference *ref, const char *name)
{
    textures.push_back(Texture(mod, ref, name));
}

#endif
