GMAC API Calls


# Memory Management #

> These functions are used to allocate, release, initialize and copy shared memroy structures.

> ## gmacMalloc ##
> > ### Syntax ###
> > `gmacError_t gmacMalloc(void **addr, size_t size)`
> > ### Description ###
> > Allocates `size` bytes of shared memory in the accelerator attached to the CPU thread calling it. This memory can be accessed by all CPU execution threads and by the accelerator that allocated it.
> > ### Parameters ###
      * `addr`: pointer to the variable where the starting memory address for the allocated memory will be stored.
      * `size`: size, in bytes, of memory to be allocated.
> > ### Return Value ###
> > This function returns `gmacSuccess` if the allocation was possible or a different return value describing the reason that prevented the memory allocation otherwise.


> ## gmacGlobalMalloc ##
> > ### Syntax ###
> > `gmacError_t gmacGlobalMalloc(void **addr, size_t size)`
> > ### Description ###
> > This function allocates shared memory in the accelerator attached to the CPU thread calling it. This memory can be accessed by all CPU execution threads and all accelerators.
> > Do not use this call unless you really need all accelerator access the allocated memory. This function requires as many accelerator memory as the requested size times the number of accelerators.
> > ### Parameters ###
      * `addr`: pointer to the variable where the starting memory address for the allocated memory will be stored.
      * `size`: size, in bytes, of the memory to be allocated.
> > ### Return Value ###
> > This function returns `gmacSuccess` is the allocation was possible or a different return value describing the reason that prevented the memory allocation otherwise.


> ## gmacFree ##
> > ### Syntax ###
> > `gmacError_t gmacFree(void *addr)`
> > ### Description ###
> > This function releases shared memory previously allocated by `gmacMalloc()` or `gmacGlobalMalloc()`.
> > ### Parameters ###
      * `addr`: starting memory address of the variable to be released.
> > ### Return Value ###
> > This function returns `gmacSuccess` is the memory was correctly release or a different return value describing the reason that prevented the operation otherwise.


> ## gmacMemset ##
> > ### Syntax ###
> > `gmacError_t gmacMemset(void *addr, int init, size_t size)`
> > ### Description ###
> > This function initializes a shared variable with a fixed value.
> > ### Parameters ###
      * `addr`: starting address of the memory to be initialized.
      * `init`: numeric value each byte will be initialized to.
      * `size`: size, in bytes, of the memory to be initialized.
> > ### Return Value ###
> > This function returns `gmacSuccess` if the operation succeeded or a different return value describing the reason that prevented the operation otherwise.


> ## gmacMemcpy ##
> > ### Syntax ###
> > `gmacError_t gmacMemcpy(void *dst, const void *src, size_t size)`
> > ### Description ###
> > This function copies the contents of memory from one accelerator to the memory belonging to a different accelerator.
> > ### Parameters ###
      * `dst`: memory address in the destination accelerator.
      * `src`: memory address in the source accelerator.
      * `size`: size, in bytes, of the memory to be copied.
> > ### Return Value ###
> > This function returns `gmacSuccess` if the contents of the memory were copied correctly, or an error describing the error otherwise.


> ## gmacSend ##
> > ### Syntax ###
> > `void gmacSend(pthread_t tid)`
> > ### Description ###
> > This function sends the accelerator currently attached to the CPU thread calling the function to the CPU execution thread identified by `tid`. After this call the calling CPU execution thread can not longer call to GMAC until a new accelerator is attached using `gmacReceive()`
> > ### Return Value ###
> > This function does not return any value.


> ## gmacCopy ##
> > ### Syntax ###
> > `void gmacSend(pthread_t tid)`
> > ### Description ###
> > This function copies the accelerator currently attached to the CPU thread calling the function to the CPU execution thread identified by `tid`.
> > ### Parameters ###
      * `tid`: Thread ID of the CPU thread receiving the accelerator.
> > ### Return Value ###
> > This function does not return any value.


> ## gmacReceive ##
> > ### Syntax ###
> > `void gmacSend()`
> > ### Description ###
> > This function deattaches the accelerator from the calling CPU execution thread and waits for a new accelerator to be sent to the thread via a call to `gmacSend()` or `gmacCopy()`.


> ### Parameters ###
    * `tid`: Thread ID of the CPU thread receiving the accelerator.
> ### Return Value ###
> This function does not return any value.


> ## gmacSendReceive ##
> > ### Syntax ###
> > `void gmacSendReceive(unsigned long tid)`
> > ### Description ###
> > This function sends the accelerator currently attached to the CPU thread calling the function to the CPU execution thread identified by `tid`.
> > ### Parameters ###
      * `tid`: Thread ID of the CPU thread receiving the accelerator.
> > ### Return Value ###
> > This function does not return any value.

# Kernel Management #


> ## gmacSetupArgument ##
> > ### Syntax ###
> > `gmacError_t gmacSetupArgument(void *addr, size_t size, size_t offset)`
> > ### Description ###
> > This function adds an argument to the next kernel call. This function is not intended to be used be programmers directly. Calls to this function usually come from the code generated by the NVIDIA compiler.
> > ### Parameters ###
      * `addr`: system memory address of the argument to be set.
      * `size`: size, in bytes, of the argument.
      * `offset`: offset, in bytes, of the argument in the kernel argument list.
> > ### Return Value ###
> > This function returns `gmacSuccess` if the argument was set correctly, or an error describing the error otherwise.



> ## gmacLaunch ##
> > ### Syntax ###
> > `gmacError_t gmacLaunch(char *kernel)`
> > ### Description ###
> > This function invokes a kernel in the accelerator. This function is not intended to be used be programmers directly. Calls to this function usually come from the code generated by the NVIDIA compiler.
> > ### Parameters ###
      * `kernel`: handler to the kernel to be invoked.
> > ### Return Value ###
> > This function returns `gmacSuccess` if the kernel was invoked correctly, or an error describing the error otherwise.


> ## gmacThreadSynchronize ##
> > ### Syntax ###
> > `gmacError_t gmacThreadSynchronize()`
> > ### Description ###
> > This function waits for the accelerator to finish all kernels previously invoked by `gmacLaunch`.
> > ### Parameters ###
> > This function receives no parameters.
> > ### Return Value ###
> > This function returns `gmacSuccess` if all previous kernels finished correctly, or an error describing the error otherwise.

# Miscellaneous Functions #


> ## gmacGetLastError ##
> > ### Syntax ###
> > `gmacError_t gmacGetLastError()`
> > ### Description ###
> > This function returns the error status from the last call to GMAC.
> > ### Parameters ###
> > This function receives no parameters.
> > ### Return Value ###
> > This function returns the error status from the last call to GMAC.


> ## gmacGetErrorString ##
> > ### Syntax ###
> > `const char *gmacGetErrorString(gmacError_t error)`
> > ### Description ###
> > This function produces an error string describing the error code received as parameter.
> > ### Parameters ###
      * `error`: error code to be converted to character string.
> > ### Return Value ###
> > This function returns a constant pointer to a char array containing the error string describing the error. This function might return NULL if the error code is invalid.


> ## gmacPtr ##
> > ### Syntax ###
> > `void *gmacPtr(void *addr)`
> > `T * gmacPtr(T *addr)`
> > ### Description ###
> > This function returns the device address associated to the shared variable at address `addr`. This function is not necessary if GMAC was compiled using the `--enable-mmap`. If host-mapped global mappings are used (`--enable-global-host` flag), this function must be used when passing pointer to global mappings, even if the `--enable-mmap` flags is enabled.
> > ### Parameters ###
      * `addr`: system memory address where the shared variable is mapped.
> > ### Return Value ###
> > Pointer to device memory where the shared variable is located.