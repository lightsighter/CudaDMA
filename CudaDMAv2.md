# CudaDMA 2.0 #

CudaDMA version 2.0 is officially released as of February 3, 2013.  Version 2.0 enhances programming with CudaDMA in the following ways:

  * Modifications to the API to support the Kepler architecture
  * A more flexible API for overlapping transfers with compute
  * Performance debugging feedback on CudaDMA instances
  * Compile-time assertions
  * Better control over caching policies

We describe each of these improvements in detail in the sections below.  CudaDMA 2.0 is completely encapsulated by the single header file cudaDMAv2.h found in the 'include' directory in the CudaDMA git repository.

## Release Notes ##

  * **CudaDMA Version 2.0 exercises a correctness bug in nvcc when compiling for arch=compute\_20 on 4.0 - 4.2 versions of nvcc only.** We've filed a bug with the compiler team and will remove this note when the bug has been fixed.  No bugs have been found when targeting arch=compute\_30 or arch=compute\_35.
  * Performance in version 2.0 is highly dependent on the register allocator in ptxas.  If it spills CudaDMA's internal buffers to local memory performance is severely degraded.  We suggest passing the '-abi=no -v' flags to ptxas and seeing if any local memory is used.  If it is, experimentation with the `BYTES_PER_THREAD` template parameter (described below) may be necessary to get data into registers.
  * There is a new example for how to use CudaDMA version 2.0 that illustrates all of the interesting buffering techniques in conjunction with all the new features in version 2.0.  The example computes a 2-dimensional stencil over a 3-dimensional space and can be found in 'src/examples/stencil' in the git repository.

## Support for Kepler ##
The release of the new Kepler architecture by NVIDIA exposed several new performance primitives.  The one most pertinent to CudaDMA was a new load instruction called LDG loads (see PTX Manual v3.1 Table 90).  LDG loads are non-coherent loads performed through the texture cache.  Loads issued through the texture cache must return inorder and have longer latency.  They also have to benefits.  LDG loads have the potential to achieve significantly higher memory bandwidth performance than loads through the L1.  More importantly, unlike loads through the L1 where the maximum number of outstanding loads per thread is limited by scoreboarding logic, a single thread can issue a virtually unlimited number of loads without blocking (more on limitations momentarily).  This is extremely beneficial for CudaDMA.  Rather than requiring 4 or more DMA warps as was necessary on Fermi to saturate memory bandwidth, on Kepler CudaDMA can saturate memory bandwidth with only 1 or 2 DMA warps per threadblock.

Getting the CUDA compiler to automatically generate LDGs is tricky, but CudaDMA makes it easy.  To automatically generate LDGs the compiler must prove that you are doing a read-only load from global memory with no aliased pointers in your environment.  This requires comprehensive `const __restrict__` annotations and some static analysis which due to its conservative nature usually fails.  To use LDGs in CudaDMA, you simply need to pass a single boolean template parameter to `execute_dma` indicating that you are loading from global memory and CudaDMA will automatically guarantee the generation of LDGs if you're compiling for GPUs with SM generation 3.5 (e.g. K20 and K20c GPUs).

```
// Overloaded execute_dma implementations
class CudaDMA {
public:
   // Old version: still available
   void execute_dma(const void *src, void *dst);
   // New version: uses LDGs if GLOBAL_LOAD is true and compiling for compute_35
   template<bool GLOBAL_LOAD>
   void execute_dma(const void *src, void *dst);
};
```

The number of LDGs outstanding for a thread at a time is actually limited by the number of registers a thread is willing to allocate for the results of outstanding loads.  Since register pressure is an application specific property, the new CudaDMA interface includes a new template parameter for all CudaDMA instances: `BYTES_PER_THREAD` which allows the programmer to control the number of registers allocated by CudaDMA for outstanding LDG loads.  This must by a multiple of the `ALIGNMENT` template parameter and defaults to `4*ALIGNMENT` which was the baked in value for Fermi (e.g. four outstanding loads at a time, which was the maximum per thread for Fermi's scoreboard logic).

The choice of `BYTES_PER_THREAD` does have an impact on performance.  Fewer outstanding loads at a time means less memory-level parallelism is being exploited and performance may suffer.  To help make this tradeoff more transparent we introduce performance debugging feedback (see below).

## Two-Phase API ##
The introduction of LDG loads also required an extension to the CudaDMA API.  Previously, a single call to 'execute\_dma' was sufficient to perform a transfer efficiently.  However the asynchronous nature of LDGs made it desirable to break transfers into two parts: issue a batch of LDGs and then at some later point synchronize waiting for all the LDGs to complete.  To facilitate this we required two method calls instead of one.  Each CudaDMA instance now supports `start_xfer_async` and `wait_xfer_finish` calls that have identical functionality to a single `execute_dma` call (in fact, `execute_dma` is now implemented by calling these two one immediately after the other).  This gives users better control over where transfers are begun and where they are finished.  This is most important for non-warp-specialized applications.  The `execute_dma` call is still available on all CudaDMA instances and has identical semantics to previous generations of CudaDMA.  Examples of using all the interfaces of CudaDMA are included in the stencil example in the git repository.
```
class CudaDMA {
public:
  // Transfer functions supported by all CudaDMA instances
  void start_xfer_async(const void *src_ptr);
  template<bool GLOBAL_LOAD>
  void start_xfer_async(const void *src_ptr);
public:
  void wait_xfer_finish(void *dst_ptr);
  template<bool GLOBAL_LOAD>
  void wait_xfer_finish(void *dst_ptr);
public:
  void execute_dma(const void *src_ptr, void *dst_ptr);
  template<bool GLOBAL_LOAD>
  void execute_dma(const void *src_ptr, void *dst_ptr);
};
```

## Performance Debugging ##
Due to the introduction of the `BYTES_PER_THREAD` parameter programmers have more control over the implementation and performance of CudaDMA instances than they did in previous versions of CudaDMA.  To make the performance of CudaDMA instances more transparent, each instance now supports a diagnostic host function that reports on the expected performance of the instance for a given configuration.
```
template<...> class CudaDMASequential {
public:
  __host__
  static void diagnose(int alignment, int bytes_per_thread, int bytes_per_elmt,
              int num_dma_threads, bool fully_templated, bool verbose = false);
};
template<...> class CudaDMAStrided {
public:
  __host__
  static void diagnose(int alignment, int bytes_per_thread, int bytes_per_elmt,
   int num_dma_threads, int num_elmts, bool fully_templated, bool verbose = false);
};
template<...> class CudaDMAIndirect {
public:
  __host__
  static void diagnose(int alignment, int bytes_per_thread, int bytes_per_elmt,
   int num_dma_threads, int num_elmts, bool fully_templated, bool verbose = false);
};
// Invocation: note assume default template arguments
CudaDMASequential<>::diagnose(...);
CudaDMAStrided<>::diagnose(...);
CudaDMAIndirect<>::diagnose(...);
```

All the arguments that impact performance are passed.  The `fully_templated` parameter indicates whether as many arguments as possible are being passed as templates or not.  The `diagnose` functions print to `stdout` and tell whether the transfers can be completed in a single pass.  If the transfers can't be done in a single pass, suggestions are given for improving performance.  An example output from a CudaDMAStrided instance is shown below:

```
//********************************************************************
//*                                                                  *
//*              Diagnostic Printing for CudaDMAStrided              *
//*                                                                  *
//********************************************************************

//  PARAMETERS
//    - ALIGNMENT:          4
//    - BYTES-PER-THREAD    4
//    - BYTES-PER-ELMT      420
//    - NUM ELMTS           30
//    - DMA THREADS         416
//    - FULLY TEMPLATED     true

//  Case: Full Elements - element sizes are sufficiently small that 1 elements
//                        can be loading by 4 warps per step.  This means there
//                        are a total of 3 elements being loaded per step.
//  TOTAL REQUIRED STEPS: 11
//  DIAGNOSIS: UN-OPTIIZED! This transfer requires multiple steps to 
//                          be performed. See recommendations below...
//  RECOMENDATIONS:
//    - Increase the number of DMA threads particpating in the transfer
//    - Increase the number of bytes available for outstanding loads
//    - Increase element size thereby loading superflous data with the benefit
//          of improving guaranteed alignment of pointers
```

## Compile-Time Assertions ##
A short-coming of previous CudaDMA implementations was that they could silently fail if dynamic checks where not intentionally enabled by a user.  These dynamic checks relied on runtime assertions which were only supported on later generation GPUs and therefore were rarely checked by much users.  The new version of CudaDMA has support for compile time assertions that require no runtime overhead and will always tell you about bugs in your code.  Compile-time assertions now check the following properties:

  * `ALIGNMENT` is one of 4-, 8-, or 16-byte aligned
  * `BYTES_PER_THREAD` is greater than zero
  * `BYTES_PER_THREAD` is a multiple of `ALIGNMENT`

While compile-time assertions don't generate the best error messages, we believe it is preferable to silent failures.  If you get a nasty looking static assertion, trace it to its source line in the code and it should be obvious about the invariant you have violated.

## Controlling Caching ##
Finally, CudaDMA exposes a feature of PTX usually only utilized by advanced CUDA programmers: the ability to control the caching effects on load and store operations.  While normally caching effects don't impact GPU performance, they can be especially important for kernels that spill registers or rely on re-use of data in the L1 or L2 caches.  CudaDMA now explicitly allows users to specify the caching effects to be applied to loads and stores performed by a CudaDMA instance.  These effects are well documented in the PTX ISA v3.1 in tables 83 and 84.
```
enum CudaDMALoadQualifier {
  LOAD_CACHE_ALL, // cache at all levels
  LOAD_CACHE_GLOBAL, // cache only in L2
  LOAD_CACHE_STREAMING, // cache at all levels, mark evict first
  LOAD_CACHE_LAST_USE, // invalidate line after use
  LOAD_CACHE_VOLATILE, // don't cache at any level
};
enum CudaDMAStoreQualifier {
  STORE_WRITE_BACK, // write-back all coherent levels
  STORE_CACHE_GLOBAL, // cache in L2 and below
  STORE_CACHE_STREAMING, // mark as evict first
  STORE_CACHE_WRITE_THROUGH, // write through L2 to system memory
};
class CudaDMA {
public:  // Additional versions of CudaDMA transfer functions
  template<bool GLOBAL_LOAD, CudaDMALoadQualifier LOAD_QUAL, CudaDMAStoreQualifer STORE_QUAL>
  void start_xfer_async(const void *src_ptr);
  template<bool GLOBAL_LOAD, CudaDMALoadQualifier LOAD_QUAL, CudaDMAStoreQualifer STORE_QUAL>
  void wait_xfer_finish(void *dst_ptr);
  template<bool GLOBAL_LOAD, CudaDMALoadQualifier LOAD_QUAL, CudaDMAStoreQualifer STORE_QUAL>
  void execute_dma(const void *src_ptr, void *dst_ptr);
};
```