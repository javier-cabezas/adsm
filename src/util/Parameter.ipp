#ifndef __PARAMS_IPP_
#define __PARAMS_IPP_

#include <cstdio>
#include <stdint.h>

namespace gmac { namespace util {

template <typename T>
static
T convert(const char * str);

template <>
inline bool convert<bool>(const char * str)
{
    return bool(atoi(str));
}

template <>
inline int convert<int>(const char * str)
{
    return atoi(str);
}

template <>
inline float convert<float>(const char * str)
{
    return atof(str);
}

template <>
inline size_t convert<size_t>(const char * str)
{
    return atol(str);
}

template <>
inline char * convert<char *>(const char * str)
{
    return (char *)str;
}

template<typename T>
inline Parameter<T>::Parameter(T *address, const char *name,
        T def, const char *envVar, uint32_t flags) :
    value(address),
    name(name),
    envVar(envVar),
    flags(flags),
    def(def)
{
    TRACE("Getting value for %s", name);
    const char *tmp = NULL;
    if(envVar != NULL &&
        (tmp = getenv(envVar)) != NULL) {
        *value = convert<T>(tmp);

        if (flags & PARAM_NONZERO &&
            *value == 0) {
            *value = def;
        } else {
            envSet = true;
        }
    }
    else {
        *value = def;
    }
    
}

template<typename T>
void Parameter<T>::print() const
{
    std::cout << name << std::endl;
    std::cout << "\tValue: " << *value << std::endl;
    std::cout << "\tDefault: " << def << std::endl;
    std::cout << "\tVariable: " << envVar << std::endl;
    std::cout << "\tFlags: " << flags << std::endl;
    std::cout << "\tSet: " << envSet << std::endl;
}

}}

#endif
