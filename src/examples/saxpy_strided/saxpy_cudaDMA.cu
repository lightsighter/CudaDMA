/*
 *  Copyright 2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/* 
 * SAXPY example code using cudaDMA library.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "cuda.h"
#include "cuda_runtime.h"

// includes, kernels
#include "saxpy_cudaDMA_kernel.cu"
#include "params.h"

#define MY_GPU_SM_COUNT 14
#define PRINT_ERRORS 1


#define CUDA_SAFE_CALL(x)					\
	{							\
		cudaError_t err = (x);				\
		if (err != cudaSuccess)				\
		{						\
			printf("Cuda error: %s\n", cudaGetErrorString(err));	\
			exit(1);				\
		}						\
	}

void process_error( cudaError_t error, char *string=0, bool verbose=false )
{
    if( cudaSuccess != error )
    {
        if( string )
            printf( "%s: ", string );
        printf( "%s\n", cudaGetErrorString( error ) );
        exit( -1 );
    }
        
    if( verbose && string )
        printf( "%s\n", string );
}


////////////////////////////////////////////////////////////////////////////////
void
computeGoldResults( float* y, float* x, float a, int num_elements) 
{
  for( unsigned int i = 0; i < num_elements; ++i) {
    y[i] = a * x[i] + y[i];
  }
}
////////////////////////////////////////////////////////////////////////////////

inline float frand() {
  return (float)rand()/(float)RAND_MAX;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

  unsigned int num_elements = CTA_COUNT * NUM_ITERS * COMPUTE_THREADS_PER_CTA;
  unsigned int mem_size = sizeof(float) * num_elements;
  // Allocate host memory
  float* h_x = (float*) malloc( mem_size );
  float* h_y = (float*) malloc( mem_size );

  // Initalize the inputs
  srand(0);
  for( unsigned int i = 0; i < num_elements; ++i) {
    h_x[i] = frand();
    h_y[i] = frand();
  }
  float a = frand();

  // Allocate device memory
  float* d_x;
  float* d_y;
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_x, mem_size));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_y, mem_size));
  // Copy host memory to device
  CUDA_SAFE_CALL( cudaMemcpy( d_x, h_x, mem_size,
			     cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy( d_y, h_y, mem_size,
			     cudaMemcpyHostToDevice) );

  // Timer setup stuff
  cudaError_t error = cudaSuccess;
  float f_time_ms=0.f;
  cudaEvent_t start, stop;
  error = cudaEventCreate( &start );
  process_error( error, "create start event" );
  error = cudaEventCreate( &stop );
  process_error( error, "create stop event" );
  
  clock_t * d_timer_vals;
  unsigned int timer_size = sizeof(clock_t) * 4 * num_elements;
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_timer_vals, timer_size));

  // Execute the kernel
  if(BYTES_PER_DMA_THREAD * DMA_THREADS_PER_LD < DMA_SZ) printf("WARNING: Transfer is too large for templated value\n");
  if(ITERS_PER_COMPUTE_THREAD * COMPUTE_THREADS_PER_CTA < DMA_SZ_IN_FS) printf("WARNING: Compute threads will not process entire transfer\n");
  if(ITERS_PER_COMPUTE_THREAD * COMPUTE_THREADS_PER_CTA > DMA_SZ_IN_FS) printf("WARNING: Compute threads will overrun shared buffer if idx not checked\n");
  if(COMPUTE_THREADS_PER_CTA > DMA_SZ_IN_FS) printf("WARNING: Compute threads on first iter will overrun shared buffer if idx not checked\n");
  printf ("Launching kernel with:\n");
  printf ("   %d total CTAs per SM\n",(CTA_COUNT/MY_GPU_SM_COUNT));
  if ( (SAXPY_KERNEL==saxpy_cudaDMA) || (SAXPY_KERNEL==saxpy_cudaDMA_doublebuffer) ) {
    printf ("   %d total threads per CTA (%d compute, %d dma)\n",THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA,DMA_THREADS_PER_CTA);
    printf ("   %d bytes per DMA thread\n",BYTES_PER_DMA_THREAD);
    printf ("   %d byte DMA transfer\n",DMA_SZ);
    printf ("   %d byte element size\n",EL_SZ);
    printf ("   %d elements\n",DMA_SZ/EL_SZ);
  } else {
    printf ("   %d total threads per CTA\n",THREADS_PER_CTA);
    printf ("   %d total warps per CTA\n",THREADS_PER_CTA/32);
  }
  cudaEventRecord( start, 0 );
  SAXPY_KERNEL<<< CTA_COUNT, THREADS_PER_CTA >>>( d_y, d_x, a, d_timer_vals);
  error = cudaThreadSynchronize();
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize(stop);
  process_error( error, "kernel" );
  error = cudaEventElapsedTime( &f_time_ms, start, stop );
  process_error( error, "get event elapsed time" );

  // Stop Timer:
  printf( "Processing time: %f (ms)\n", f_time_ms);
  printf( "Bytes Processed: %d KB\n", static_cast<int>(static_cast<float>(mem_size) / 1024.0 ));

  // Allocate mem for the device results on host side
  float* h_results = (float*) malloc( mem_size );
  clock_t* h_timer_vals = (clock_t*) malloc (timer_size);
    
  // copy result from device to host
  CUDA_SAFE_CALL( cudaMemcpy( h_results, d_y, mem_size, 
			     cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_timer_vals, d_timer_vals, timer_size, 
			     cudaMemcpyDeviceToHost) );

#ifdef TIMERS_ON
  unsigned int warp0_time = (unsigned int) h_timer_vals[0];
  for (unsigned int i = 0; i < num_elements; i+=32) {
    //unsigned int diffclock = (unsigned int) h_timer_vals[num_threads+i] - (unsigned int) h_timer_vals[i];
    //printf("Warp %d iter cycles = %d\n",i,diffclock);
    printf("Warp %d clock1 = %d\n",i,(unsigned int) h_timer_vals[i] - warp0_time);
    printf("Warp %d clock2 = %d\n",i,(unsigned int) h_timer_vals[num_threads+i] - warp0_time);
    printf("Warp %d clock3 = %d\n",i,(unsigned int) h_timer_vals[2*num_threads+i] - warp0_time);
    printf("Warp %d clock4 = %d\n",i,(unsigned int) h_timer_vals[3*num_threads+i] - warp0_time);
  }
#endif

  // Compute and compare host reference results
  computeGoldResults( h_y, h_x, a, num_elements );
  bool res;
  res = true;
  for (unsigned int i = 0; i < num_elements; ++i) {
    // Not sure, but I think you can have rounding errors because GPUs support FMAs...
    if ((h_y[i]-h_results[i])>0.000001) {
      res = false;
#ifdef PRINT_ERRORS
      printf("ERROR: host y[%d] = %f\tdevice y[%d]=%f\tdifference = %f\n",i,h_y[i],i,h_results[i],h_results[i]-h_y[i]);
#endif
    } else {
      //printf(":)_:): host y[%d] = %f\tdevice y[%d]=%f\n",i,h_y[i],i,h_results[i]);
    }
  }
  if (res) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
    printf("alpha = %f\n",a);
  }

  // Report bandwidths
  double d_log2_size = log2( static_cast<double> (mem_size/4) );
  printf( "Vector Dimension:  2^%.1f\n", d_log2_size);
  float f_bw = static_cast<float>(3*mem_size) / f_time_ms / static_cast<float>(1000000.0);
  printf( "Bandwidth: %f GB/s\n", f_bw);
  float f_gflops = static_cast<float>(2*(mem_size/4)) / f_time_ms / static_cast<float>(1000000.0);
  printf( "Performance: %f GFLOPS\n", f_gflops);
    
  // cleanup memory
  free( h_x );
  free( h_y );
  free( h_results );
  free( h_timer_vals );
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_y));

  cudaThreadExit();
}

