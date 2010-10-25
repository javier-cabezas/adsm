#ifndef __PARAMS_IPP_
#define __PARAMS_IPP_

#include <cstdio>

#if defined(__GNUC__)
#	define GETENV getenv
#elif defined(_MSC_VER)
#	define GETENV gmac_getenv
static inline const char *gmac_getenv(const char *name)
{
	static char buffer[512];
	size_t size = 0;
	if(getenv_s(&size, buffer, 512, name) != 0) return NULL;
	if(strlen(buffer) == 0) return NULL;
	return (const char *)buffer;
}
#endif

namespace gmac { namespace util {

template <typename T>
static
T convert(const char * str);

template <>
inline bool convert<bool>(const char * str)
{
    return bool(atoi(str) != 0);
}

template <>
inline int convert<int>(const char * str)
{
    return atoi(str);
}

template <>
inline float convert<float>(const char * str)
{
    return (float)atof(str);
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

template <>
inline const char * convert<const char *>(const char * str)
{
    return str;
}

template<typename T>
inline Parameter<T>::Parameter(T *address, const char *name,
        T def, const char *envVar, uint32_t flags) :
    value(address),
    def(def),
    name(name),
    envVar(envVar),
    flags(flags)
{
    const char *tmp = NULL;
    if(envVar != NULL &&
        (tmp = GETENV(envVar)) != NULL) {
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
