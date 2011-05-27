#include <gmac/opencl.h>

. . .

    oclError_t error_code;
    error_code = oclCompileSourceFile(kernel_file);
    if(error != oclSuccess) error(error_code);
