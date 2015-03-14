# Getting Started with CudaDMA #

## Download ##
To get started with CudaDMA first checkout the code from our git repository using the following command.

```
git clone https://code.google.com/p/cudadma/
```

This will give you a read-only copy of the library.

## Requirements ##
To use CudaDMA you must have a version of `nvcc` that supports inline PTX statements.  The version of `nvcc` distributed with CUDA 4.0 supports inline PTX statements and has been used to test all CudaDMA code.

In terms of hardware requirements, CudaDMA can be used on all CUDA capable devices, but is only supported in restricted forms on earlier devices.  The CudaDMA [programming model](ProgrammingModel.md) describes the two ways that CudaDMA can be employed.  Using explicit DMA warps is only supported on devices of compute capability 2.0 and later as they contain the fine-grained synchronization primitives necessary for warp specialization.  Using CudaDMA without explicit DMA warps can be done on all CUDA capable devices.

## Using CudaDMA ##

There are two top-level directories in the CudaDMA repository: `include` and `src`.  The `include` directory contains the `cudadma.h` header file.  All of the code required for using the CudaDMA library is contained in this file.  To use any of the objects in the CudaDMA library simply include this header file at the top of any application code.  All the source for the CudaDMA objects are contained in the header file since linking is not supported in `nvcc` for device code.

In addition to the CudaDMA header file in the `include` directory, there are a combination of example codes and test benches in the `src` directory.  The `src/tests` directory contains test benches for each of the CudaDMA objects.  The `src/examples` directory contains programs that demonstrate how to use CudaDMA instances.

  * `saxpy` - performing saxpy with [CudaDMASequential](CudaDMASequential.md)
  * `saxpy_strided` - performing saxpy with [CudaDMAStrided](CudaDMAStrided.md)
  * `sgemv` - performing sgemv with both [CudaDMASequential](CudaDMASequential.md) and [CudaDMAStrided](CudaDMAStrided.md) in the same kernel

To illustrate the workings of an actual application the next section describes the `saxpy` example using [CudaDMASequential](CudaDMASequential.md) at a high level.

## Example Application: SAXPY ##

The `saxpy` computation takes two vectors of the same size, `x` and `y`, and a constant `a` and performs the computation `y[i] += a*x[i]` for all `i`.  To perform this computation each threadblock is responsible for performing the computation for a subset of the elements.  If the number of elements that a threadblock is responsible for is larger than the number of threads in the threadblock then the threadblock will contain a loop to iterate over the elements in its subset.

While this computation can be expressed easily in CUDA as a streaming computation without using shared memory, we will demonstrate it using shared memory.  By using explicit warp specialization as described in the CudaDMA [programming model](ProgrammingModel.md) we will have DMA threads first move the `x` and `y` data into shared memory and then have the compute threads process it out of shared memory.  Even though this may seem to be less efficient as it requires an extra exchange of data through shared memory, we have found applications where the improvement in memory bandwidth performance from using CudaDMA objects outweighs the cost of the data exchange.

The kernel for performing `saxpy` with CudaDMA can be seen below.

```
__global__
void saxpy_cudaDMA(float *y, float *x, float a)
{
  __shared__ float sdata_x[NUM_COMPUTE_THREADS];
  __shared__ float sdata_y[NUM_COMPUTE_THREADS];

  // DMA object for loading buffer for x data
  cudaDMASequential<16, NUM_COMPUTE_THREADS*sizeof(float), DMA_THREADS_PER_LD>
    dma_ld_x(0, NUM_COMPUTE_THREADS, NUM_COMPUTE_THREADS);
  // DMA objects for loading buffer for y data
  cudaDMASequential<16, NUM_COMPUTE_THREADS*sizeof(float), DMA_THREADS_PER_LD>
    dma_ld_y(1, NUM_COMPUTE_THREADS, NUM_COMPUTE_THREADS+DMA_THREADS_PER_LD);

  if (dma_ld_x.owns_this_thread())
  {
    // DMA threads for loading data for x vector
    // Iterate over the elements for a block
    for (int i = 0; i<(ELMT_PER_BLOCK/NUM_COMPUTE_THREADS); i++)
    {
      int idx = blockIdx.x*ELMT_PER_BLOCK + i*NUM_COMPUTE_THREADS;
      dma_ld_x.execute_dma(&x[idx], sdata_x);
    }
  }
  else if (dma_ld_y.owns_this_thread())
  {
    // DMA threads for loading data for y vector
    for (int i = 0; i<(ELMT_PER_BLOCK/NUM_COMPUTE_THREADS); i++)
    {
      int idx = blockIdx.x*ELMT_PER_BLOCK + i*NUM_COMPUTE_THREADS;
      dma_ld_y.execute_dma(&y[idx], sdata_y);
    }
  }
  else
  {
    // Compute threads for processing both buffers
    float tmp_x, tmp_y;
    int i;
    // Preamble (start DMA operations}
    dma_ld_x.start_async_dma();
    dma_ld_y.start_async_dma();
    for (i = 0; i<(ELMT_PER_BLOCK/NUM_COMPUTE_THREADS-1); i++)
    {
      // Get the x value from the shared buffer and start the next load
      dma_ld_x.wait_for_dma_finish();
      tmp_x = sdata_x[threadIdx.x];
      dma_ld_x.start_async_dma();
      // Get the y value from the shared buffer and start the next load
      dma_ld_y.wait_for_dma_finish();
      tmp_y = sdata_y[threadIdx.x];
      dma_ld_y.start_async_dma();
      // Perform the computation
      int idx = blockIdx.x*ELMT_PER_BLOCK + i*NUM_COMPUTE_THREADS + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
    }
    // Postamble (last iteration)
    dma_ld_x.wait_for_dma_finish();
    tmp_x = sdata_x[threadIdx.x];
    dma_ld_y.wait_for_dma_finish();
    tmp_y = sdata_y[threadIdx.x];
    int idx = blockIdx.x*ELMT_PER_BLOCK + i*NUM_COMPUTE_THREADS + threadIdx.x;
    y[idx] = a * tmp_x + tmp_y;
  }
}
```

At a high level this application works by first declaring two shared memory buffers, `sdata_x` and `sdata_y`.  For each of these buffers we declare an instance of [CudaDMASequential](CudaDMASequential.md) to manage the transfers into these buffers.

We then split the threadblock's threads into compute threads and two different sets of DMA threads.  This is done by the `if-else` blocks that query the CudaDMA objects using the `owns_this_thread` function.  Each of the DMA threads will then repeatedly load the shared buffers when prompted by the compute threads by executing the `execute_dma` calls.  To indicate when buffers are loaded, the compute threads perform the calls `start_async_dma` and `wait_for_dma_finish` on the CudaDMA objects.  The semantics of these calls are described in more detail in the description of the CudaDMA [API](Interface.md).