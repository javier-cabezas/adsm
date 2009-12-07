#ifndef __PARAMS_IPP_
#define __PARAMS_IPP_

#include <cstdio>

template <typename T>
static
T convert(char * str);

template <>
bool convert<bool>(char * str)
{
    return bool(atoi(str));
}

template <>
int convert<int>(char * str)
{
    return atoi(str);
}

template <>
float  convert<float>(char * str)
{
    return atof(str);
}

template <>
size_t convert<size_t>(char * str)
{
    return atol(str);
}

template <>
char * convert<char *>(char * str)
{
    return str;
}

struct ParamDescriptor {
    const char * env;
    uint32_t flags;
    bool envSet;
    void (*print)();
    void (*print_default)();
};

extern std::map<const char *, ParamDescriptor> params;

template <typename T>
static
void paramCheckAndSet(T * v, T defaultValue, const char * name, void (*print)(), void (*print_default)(), const char * env = NULL, uint32_t flags = 0)
{
    ParamDescriptor desc;

    desc.env    = env;
    desc.flags  = flags;
    desc.envSet = false;
    desc.print  = print;
    desc.print_default = print_default;

    char * __tmp;
    if (env && 
        (__tmp = getenv(env)) != NULL) {
        printf("Hola: %s\n", __tmp);
        T val = convert<T>(__tmp);

        if (flags & PARAM_NONZERO &&
            val == 0) {
            *v = defaultValue;
        } else {
            *v = val;
            desc.envSet = true;
        }
    } else {
        *v = defaultValue;
    }

    params[name] = desc;
}

#endif
