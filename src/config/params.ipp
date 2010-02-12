#ifndef __PARAMS_IPP_
#define __PARAMS_IPP_

#include <cstdio>
#include <stdint.h>

namespace gmac { namespace params {

template <typename T>
static
T convert(const char * str);

template <>
bool convert<bool>(const char * str)
{
    return bool(atoi(str));
}

template <>
int convert<int>(const char * str)
{
    return atoi(str);
}

template <>
float  convert<float>(const char * str)
{
    return atof(str);
}

template <>
size_t convert<size_t>(const char * str)
{
    return atol(str);
}

template <>
char * convert<char *>(const char * str)
{
    return (char *)str;
}

inline std::vector<Root *> &Root::params()
{
    if(__params == NULL) __params = new std::vector<Root *>();
    return *__params;
}

template<typename T>
void Parameter<T>::value(std::string label, std::ostream &os)
{
    os << label << __value;
}

template<typename T>
void Parameter<T>::def(std::string label, std::ostream &os)
{
    os << label << __def;
}

template<typename T>
Parameter<T>::Parameter(const char *name, T def, const char *env, uint32_t flags) :
    Root(name, env, flags), __def(def)
{
    const char *tmp = NULL;
    if(env != NULL &&
        (tmp = getenv(env)) != NULL) {
        __value = convert<T>(tmp);

        if (flags & PARAM_NONZERO &&
            __value == 0) {
            __value = __def;
        } else {
            envSet = true;
        }
    }
    else {
        __value = __def;
    }
    
}

template<typename T>
T Parameter<T>::value() const {
    return __value;
}


}}
#endif
