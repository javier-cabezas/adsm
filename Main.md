# ADSM #

Asymmetric Distributed Shared Memory is a memory and execution model for programming systems using both, general purpose processors and accelerators.

# Supported Systems #
  * GNU/Linux (i686 and amd64) running CUDA 2.x and CUDA 3.x
  * Mac OS X 10.6 (Snow Leopard) 32-bit (see MacOs for details)

# Overview #
## Memory Model ##

ADSM implements and asymmetric distributed shared memory model. Each user process has an unique virtual address space, which encompasses system memory and GPU memories. However, ADSM assigns asymmetric roles and data visibility to CPUs and GPUs. CPUs can access any memory address within the process virtual address space (whether the virtual memory address is mapped to system or GPU physical memory addresses), but GPUs are constrained to access those virtual memory address that are mapped to its physical memory address space. Note that both, CPU and GPU can use the very same virtual memory address to access data in the accelerator physical memory. This memory model leads to the following indications when accessing memory:
  * Code running in the CPU can access any program variable
  * Code running in the GPU can only access to those variables hosted by its memory. Note that code running in one GPU **can not** access to variables hosted by other GPU memory.

## Execution Model ##

The ADSM execution model extends the traditional _execution thread_ with an **GPU execution flow**. In the ADSM model, each execution thread has an associated GPU execution flow, which represents a GPU in the system. Any operation executed by one CPU execution thread involving a GPU (e.g. allocating GPU memory or launching a kernel) is performed over the associated GPU execution flow. Thus, a kernel invoked from a given CPU thread **MUST** only access the memory allocated from that CPU thread, otherwise the GPU code will cause a memory protection fault (`Xid` messages from the NVIDIA driver).

An application can create as many CPU execution threads (e.g. using `pthread_create()`) as required, and each execution thread will be assigned with a new GPU execution flow. There is no limit on the number of GPU execution flows: an application is allowed to have more GPU execution flows than GPUs are physically attached to the system.

ADSM allows GPU execution flow migration: one CPU thread can send its GPU execution flow to other CPU thread. This is a convenient way of allowing other CPU threads to launch kernels at the GPU that require data produced by GPU kernels previously launched by a different CPU execution thread.