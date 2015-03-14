# (DECREMENTED!) #



# Introduction #

The cudaDMA C++ library wraps efficient implementations of common usage patterns for moving data between shared memory and global memory of CUDA-enabled GPUs.  The class implementation assumes that a thread block will be split into **compute** and **DMA** threads.  DMA threads transfer data between global and shared memory.  Compute threads process data that has been transferred into shared memory.

The library API can be divided into two major sections, (1) declaration and (2) launch and synchronization.  Both are called from within CUDA kernels.  Typically declaration would happen outside of a main loop whereas launch and synchronization be called from within a main loop.

[The cudaDMA class declaration](Documentation#cudaDMA_Declaration.md) describes the memory access pattern for a DMA and indicates how threads in a thread block will be divided between communication and computation tasks.  This is done within a kernel by declaring a cudaDMA object with the appropriate template parameters and constructor arguments.  This component is not necssarily meant to be fast but is designed to pre-compute a number of per-thread variables as part of the class.

[Launch and Synchronization functions](Documentation#Launch_and_Synchronization.md)
provide mechanisms for launching DMAs to copy data between global and shared memory and for synchronizing between communication (DMA) and computation tasks within a kernel.  Copying and synchronizing is done by calling the appropriate member functions of a previously-declared cudaDMA class instance.

# Description #

## cudaDMA Declaration ##

Although the cudaDMA base class should not be instantiated directly in a kernel (the access-pattern-specific derived classes should be used instead), the following cudaDMA base class parameters apply to all access patterns:

  * `const int dmaID`
    * Unique integer identifier for this DMA pattern
    * Must be unique in the sense that no other concurrent cudaDMA class can have the same identifier.
    * Corresponds to a synchronization ID so using a static compile-time constant for this value leads to more efficient code.

  * `const int num_dma_threads`
    * The number of DMA threads allocated to this cudaDMA class.

  * `const int num_compute_threads`
    * The number of compute threads that will synchronize with the DMA threads for this cudaDMA class.

  * `const int dma_threadIdx_start`
    * The starting threadIdx.x thread ID for the DMA threads

### Sequential Access Patterns ###

Sequential DMA access patterns can be declared in a kernel by calling the following constructor:

```
template <int BYTES_PER_THREAD, int ALIGNMENT>
class cudaDMASequential : public cudaDMA {
  __device__ cudaDMASequential (const int dmaID,
				const int num_dma_threads,
				const int num_compute_threads,
				const int dma_threadIdx_start)
```

Declaring a `cudaDMASequential` class indicates that a block of data of size `(4*ALIGNMENT*num_dma_threads)` should be transferred from the source base address to a destination base address whenever a execute\_dma() member function is called later on the `cudaDMASequential` class.  This constructor assumes that BYTES\_PER\_THREAD==4\*ALIGNMENT and is an optimized version of the cudaDMASequential class. If the size of the transfer is not divisible by `4*ALIGNMENT*num_dma_threads` bytes, the following constructor must be used:

```
template <int BYTES_PER_THREAD, int ALIGNMENT>
class cudaDMASequential : public cudaDMA {
  __device__ cudaDMASequential (const int dmaID,
				const int num_dma_threads,
				const int num_compute_threads,
				const int dma_threadIdx_start, 
				const int sz)

```

In this case, the size is declared explicitly and the programmer must ensure that it is a multiple of ALIGNMENT bytes and does not exceed `(BYTES_PER_THREAD*num_dma_threads`, otherwise results will be undefined.

Note that `BYTES_PER_THREAD` is a template parameter so must be a static compile-time constant.  If the size of the transfer is determined dynamically at runtime, the second constructor must be used where `(BYTES_PER_THREAD*num_dma_threads)` can be thought of as the maximum transfer size.  For all of the other arguments, although they are not required to be statically determined at compile time, constants should be used wherever possible for optimal performance.

The following example describes a 64-byte-per-thread 32-dma-thread (2 KB) DMA from global to shared memory using threadIdx.x in the range [512:544) and with a DMA ID of '1':
```
cudaDMASequential<64,16> my_dma (1, 32, 512, 512);
```

### Strided Access Patterns ###

Strided DMA access patterns can be declared in a kernel by calling the following constructor:
```
template <int BYTES_PER_THREAD>
class cudaDMAStrided : public cudaDMA {
  __device__ cudaDMAStrided (const int dmaID,
                             const int num_dma_threads,
                             const int num_compute_threads,
                             const int dma_threadIdx_start,
                             const int el_sz,
                             const int el_stride)
```

Declaring a `cudaDMAStrided` class indicates that a strided block of data of total size `(BYTES_PER_THREAD*num_dma_threads)` should be transferred from the source base address to a destination base address whenever the `execute_dma()` member function is called later on the `cudaDMAStrided` class. However, unlike `cudaDMASequential`, the user specifies an element size (`el_size`) and a element source stride (`el_stride`).

This use case assumes that the total number of strided elements is `num_dma_threads*(BYTES_PER_THREAD/el_sz)` and that the size of each element is divisible by `4*num_dma_threads` bytes. Furthermore, `BYTES_PER_THREAD` must be a compile time constant. If any of those conditions are unacceptable, the following constructor must be used:

```
template <int BYTES_PER_THREAD>
class cudaDMAStrided : public cudaDMA {
  __device__ cudaDMAStrided (const int dmaID,
                             const int num_dma_threads,
                             const int num_compute_threads,
                             const int dma_threadIdx_start,
                             const int el_sz,
                             const int el_cnt,
                             const int el_stride)
```

In this case, both the element size and the element count are declared explicitly. The programmer must ensure that the element size is a multiple of 4 bytes and that `el_sz*el_cnt` does not exceed `BYTES_PER_THREAD*num_dma_threads`, otherwise results will be undefined.

In this constructor, `(BYTES_PER_THREAD*num_dma_threads)` can be thought of as the maximum transfer size.  For all of the other arguments, although they are not required to be statically determined at compile time, constants should be used wherever possible for optimal performance.

### Indirect Access Patterns ###

Indirect DMA access patterns can be declared in a kernel by calling the following constructor:
```
template <type T>
class cudaDMAIndirect : public cudaDMA {
  __device__ cudaDMAIndirect (const int dmaID,
                             const int num_dma_threads,
                             const int num_compute_threads,
                             const int dma_threadIdx_start,
                             const int el_cnt,
                             const T** idx_array_base)
```
el\_sz is implicitly sizeof(T).

### Custom Access Patterns ###

Custom DMA access patterns can be declared in a kernel by calling the following constructor:

```
class cudaDMACustom : public CUDADMA_BASE {
  __device__ cudaDMACustom (const int dmaID,
				const int num_dma_threads,
				const int num_compute_threads,
				const int dma_threadIdx_start)
```

Declaring a `cudaDMACustom` class indicates that the user will be supplying their own implementation of the data transfer functionality. In addition to writing the code executed by the DMA warps, the user must manage the synchronization with the compute warps. DMA warp synchronization functions are provided by the class, see below.

## Launch and Synchronization ##

  * `void cudaDMA::start_async_dma ( )`
    * Non-blocking launch of streaming copy to/from this object's buffer
    * Used by compute threads

  * ` void cudaDMA::wait_for_dma_finish ( ) `
    * Block until the streaming copy associated with this object's buffer has completed
    * Used by compute threads

  * `void cudaDMACustom::wait_for_dma_start ( )`
    * Wait for the signal indicating the transfer associated with this object's buffer should begin
    * Used by dma threads performing a custom copy pattern

  * `void cudaDMACustom::finish_async_dma( )`
    * Signal that the transfer associated with this object's buffer has completed
    * Used by dma threads performing a custom copy pattern

  * ` void cudaDMA::execute_dma ( void* src_ptr, void* dst_ptr ) `
    * Launches the precomputed transfer using the specified pointers to source and destination memory
    * Subsumes wait\_for\_start() and and finish\_async\_dma() calls

## Other Functions ##

  * `bool cudaDMA::owns_this_thread( ) `
    * Determines whether this thread is a DMA thread associated with this object's buffer


# Example Code #

## Sequential Cases ##

### A SAXPY kernel using cudaDMA to stage data through shared ###

This kernel executes a saxpy function Y = a\*X + Y but uses cudaDMA to stage the load of the X and Y vectors through shared memory.  Note that it is a bit silly to implement saxpy by staging data through shared using cudaDMA since there is no read reuse on the X or Y vectors in the kernel. However, saxpy makes an instructive an example since it is simple and easy to understand.

```
/*
 * This version of saxpy uses cudaDMA for DMAs into shared
   (but would require 2 CTAs/SM for double buffering).
 */
__global__ void saxpy_cudaDMA ( float* y, float* x, float a, clock_t * timer_vals) 
{
  __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
  __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];

  cudaDMASequential<BYTES_PER_DMA_THREAD>
    dma_ld_x_0 (1, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA, DMA_SZ);
  cudaDMASequential<BYTES_PER_DMA_THREAD>
    dma_ld_y_0 (2, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD, DMA_SZ);

  int tid = threadIdx.x ;

  if ( tid < COMPUTE_THREADS_PER_CTA ) {
    unsigned int idx;
    float tmp_x;
    float tmp_y;
    
    // Preamble:
    dma_ld_x_0.start_async_dma();
    dma_ld_y_0.start_async_dma();
    for (int i=0; i < NUM_ITERS; ++i) {
      dma_ld_x_0.wait_for_dma_finish();
      tmp_x = sdata_x0[tid];
      dma_ld_x_0.start_async_dma();
      dma_ld_y_0.wait_for_dma_finish();
      tmp_y = sdata_y0[tid];
      dma_ld_y_0.start_async_dma();
      idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
    }

  } else if (dma_ld_x_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; ++j) {
      // idx is a pointer to the base of the chunk of memory to copy
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_x_0.execute_dma( &x[idx], sdata_x0 );
    }
    dma_ld_x_0.wait_for_dma_start();
  } else if (dma_ld_y_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; ++j) {
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_y_0.execute_dma( &y[idx], sdata_y0 );
    }
    dma_ld_y_0.wait_for_dma_start();
  }
}
```

## Strided Cases ##

## Indirect Cases ##

# Programmer Best Practices #

## Restrictions ##

  * Only works on multiple-of-32 num\_dma\_threads and num\_compute\_threads
  * Only works with byte-aligned addressing and multiples of 4-byte sizes.

## Performance Optimizations ##

  * Use static compile-time constants wherever possible
  * Highest GB/s/thread usually achieved with 64 B/thread
    * 64 B/thread has most outstanding memory refs per thread and efficient synchronization
    * <64 B/thread has fewer outstanding memory refs per thread but efficient synchronization
    * >64 B/thread reverts to a slightly less efficient synchronization

# Feature Requests and Future Work #
  * 1. **DONE** Add optimization for strided cases with small element sizes based on current SGEMV implementation
    * **DONE** Parameterization of constructors and factoring of class hierarchy for this versus the indirect classes
    * **DONE** Implement SGEMV code using built in class.
  * 1. **DONE** FFT benchmark (Mike)
  * 1. Dense GEMM examples
  * 2. **DONE** Implement a non-synchronizing version of execute\_dma() for people who want to use the library for the access pattern encapsulation but don't want to program using the separate compute/DMA thread approach.
  * 3. **DONE** More rigorous functional testing of sequential and strided cases
  * 3. **DONE** Determine API for indirect cases
  * 3. **DONE** Implement indirect cases (ideally fully featured and fast enough to replace some cudaDMACustom classes).
  * 3. **DONE** Make sure documentation matches all of our code so we can reference website (Mike)
  * 3. **DONE** Write a brief user manual (Mike)
  * 4. Add asserts to make it easier to debug programmer errors when using cudaDMA.
    * Determine nvcc compile time and runtime assert capabilities
    * Determine best way to have debug and real versions of code (if compiler)
  * 4. Add a version of strided that can be templated off a struct type (for AoS to SoA)
  * 4. Improved language mechanisms/approaches for splitting compute/DMA threads
    * One possible approach: does it make sense to create an enqueue\_dmas() function outside the main loop to encapsulate the looping that goes on in the DMA threads?