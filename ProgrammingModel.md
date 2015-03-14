# Introduction #

There are two different approaches to using the CudaDMA API.  The original design of CudaDMA assumed that a set of explicit DMA warps would be created in addition to a base set of compute warps.  The extra DMA warps would be _specialized_ to emulate the behavior of a DMA engine.  Our experiments have shown that there are many applications that can benefit from warp specialization.  However, as we have had users begin using the API, many have stated that the programmability benefits of the API alone would be useful, independent of warp specialization.  We have therefore introduced a second model of using CudaDMA that does not require warp specialization.  We now cover both of these approaches in further detail.

## CudaDMA Using Warp Specialization ##

The primary motivation behind CudaDMA is to create specialized warps in a kernel to emulate the behavior of a DMA engine.  By separating out DMA warps that are responsible for memory operations from compute warps that are responsible for performing computation on data in shared memory, we can untangle many of the performance penalties associated with the in-order instruction issue nature of GPUs.

To perform warp specialization we make use of the CudaDMA [API](Interface.md).  For every shared memory buffer that we need to have loaded we declare a CudaDMA object.  The CudaDMA object is then responsible for managing the DMA threads that will be used loading the shared memory buffer.  Multiple CudaDMA objects can be used in the same kernel for managing different buffers.

To illustrate how a CudaDMA kernel is set up we now will walk through a simple example.  Below we have code for an example CudaDMA kernel.
```
__global__
void cuda_dma_example_kernel(float *g_data)
{
  // Global buffer
  __shrared__ float buffer[NUM_ELMTS];
  // CudaDMA object
  cudaDMA dma_ld<true/*warp specialization*/>
    (0,NUM_DMA_THREADS,NUM_COMPUTE_THREADS, NUM_COMPUTE_THREADS);

  if (dma_ld.owns_this_thread())
  {
    // DMA warps
    for (int i=0; i<NUM_ITERS; i++) 
    {
      // Contains internal synchronization
      // wait_for_dma_start
      // Load data from global to shared
      dma_ld.execute_dma(g_data,buffer);
      // finish_async_dma
    }
  }
  else
  {
    // Compute warps
    for (int i=0; i<NUM_ITERS; i++)
    {
      dma_ld.start_async_dma();
      // Do any extra work
      dma_ld.wait_for_dma_finish();
      process_buffer(buffer);
    }
  }
}
```

This example code begins by declaring a shared memory buffer at the scope of the entire kernel.  This buffer must be declared at the entire kernel scope as it will be both loaded by the DMA threads as well as read from by the compute threads.  After the buffer declaration, we have declared a CudaDMA object to manage the buffer.  This object is passed a unique ID, the number of DMA threads that it will manage, the number of compute threads to synchronize with, and the index of the first thread DMA thread (immediately after the compute threads by convention).  In addition to this we pass a boolean template parameter to the CudaDMA object to indicate that warp specialization will be used.  The details of these parameters are more fully described in the CudaDMA [API](Interface.md) description.

After declaring a CudaDMA object, we can now split the DMA threads from the compute threads.  To do this we use an `if-else` statement and ask the CudaDMA object to identify the threads that it owns.  DMA threads take the `if` branch while compute threads take the `else` branch.

In many of our applications we have found that the shared memory buffer is filled multiple times by a single threadblock.  In this example we therefore introduce a loop to be performed `NUM_ITERS` times.  This is not necessary for achieving high performance.  CudaDMA has shown to be effective even when each CudaDMA object is used only once per threadblock.

To execute each iteration of the loop, the DMA threads will repeatedly call the `execute_dma` operation to load data into the shared memory buffer.  The `execute_dma` call contains internal synchronization methods (indicated by comments in the code).   These synchronization points wait for the compute threads to indicate that the DMA operation should commence and then indicate to the compute threads that the buffer is full.

The corresponding calls in the compute threads show that the compute threads can indicate to the DMA threads when to begin a transfer (`begin_async_dma`) and then to wait for the transfer to be complete (`wait_for_dma_finish`).  Note that the call to `begin_async_dma` is non-blocking which enables the compute threads to perform additional work while waiting for the transfer to complete.  This feature is especially useful when writing CudaDMA code with different [buffering](Buffering.md) techniques.  After the transfer is complete, the compute threads can then process the buffer.

There are several important advantages to note about this style of programming.  First, the programmer has declared a CudaDMA object to load the shared memory buffer without having to say in any way how this load is performed.  All details about how the load occurs are wrapped by the CudaDMA object.  Second, there are two completely separate instruction streams for the DMA threads and the compute threads.  In the DMA thread instruction stream, most operations will be memory operations with very few computer operations.  In the compute instruction stream the opposite is true.  By separating compute and memory instruction streams we avoid stalls that could result due to the in-order nature of GPUs when one type of instruction saturates the machine.  In addition, we've allowed both the programmer and the compiler to optimize each instruction stream in isolation without have to reason about computation and memory accesses in the same stream.  Finally, the use of fine-grained synchronization primitives have enabled use to overlap computation and memory accesses through non-blocking synchronization calls.

## CudaDMA Without Warp Specialization ##

In addition to using the CudaDMA API with separate DMA threads, it can also be used as a means to describe transfer patterns independent from their implementation.  To make this possible we illustrate an example kernel.

```
__global__
void cuda_dma_kernel_no_dma_warps(float *g_data)
{
  // Shared memory buffer
  __shared__ float buffer[NUM_ELMTS];
  // CudaDMA object to manage buffer
  cudaDMA dma_ld(0,NUM_THREADS,
      NUM_THREADS,0);
 
  for (int i=0; i<NUM_ITERS; i++)
  {
    dma_ld.execute_dma_no_sync(g_data,buffer);
    __syncthreads();
    process_buffer(buffer);
    __syncthreads();
  }
}
```

Similar to the case with warp specialization, we declare a shared memory buffer and a CudaDMA object to manage it.  When the CudaDMA object is declared, however, we now specify that all of the threads in the threadblock will be used as DMA threads.  The last parameter says that the first thread used by the CudaDMA object will be thread zero.  The first and third parameters are now irrelevant since there will be no synchronization performed by the CudaDMA object.

When we enter the loop to perform computation, we issue an `execute_dma_no_sync` operation which will transfer the data using all the threads in the threadblock in the same way that explicit DMA threads would have done.  The only difference is that the `execute_dma_no_sync` operation performs no synchronization.  We therefore issue a sync threads call after the execute call to ensure that the transfer is complete.  We can then process the buffer and issue another syncthreads call to ensure that the buffer has been processed prior to starting the next iteration of the loop.