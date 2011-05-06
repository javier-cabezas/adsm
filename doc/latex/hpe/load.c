    oclError_t error_code;
    error_code = __oclPrepareCLCodeFromFile(kernel_file);
    if(error != oclSuccess) error(error_code);
