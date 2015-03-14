# Using Valgrind #

We are quite concerned about avoiding memory leaks in the GMAC code. Due to the nature of GMAC, memory leaks might decrease the amount of available system and GPU memory easily. Hence, before releasing a new version of GMAC we run it through [[http://valgrind.org](valgrind.md)] to detect any potential memory leak.

Valgrind makes really nasty things, such as re-ordering memory accesses, to speed-up the application execution. Typically, tricks played by valgrind does not affect applications, but GMAC relies on ordered memory accesses to work properly. Hence, we must use the `--vex-iropt-precise-memory-exns=yes` flag when executing GMAC. For instance, to check for memory leaks on the `gmacVecAdd` test, you have to run
```
valgrind --vex-iropt-precise-memory-exns=yes ./tests/gmacVecAdd
```
Please, before submitting a patch, run your code through valgrind and ensure that you did not introduced any memory leak.

Valgrind does not seem to track calls to mmap() to request anonymous memory. Thus, if you compiled GMAC using the `--enable-mmap` flags, you will get many errors regarding to read/write accesses to non allocated memory. Do not worry about them :-). We are also getting a memory leak regarding the page table (`VM.ipp` file), but it is being released when the page table is destroyed. If you have any hint about this strange memory leak, let us know, otherwise just ignore it, you will be happier.