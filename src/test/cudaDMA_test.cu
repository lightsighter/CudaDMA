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

/* Software DMA project
*
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
//#include "cuPrintf.cu"
#include <cudaDMA_test_kernel.cu>
//#include <cudaDMA_ref_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
void
computeGoldResults( float4* reference, float4* idata, const unsigned int num_elements, const unsigned int num_iters) 
{
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
      reference[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for( unsigned int i = 0; i < num_iters; i++) 
    {
      for( unsigned int j = 0; j < num_elements; j++) {
	reference[j].x = reference[j].x + idata[i*num_elements+j].x;
	reference[j].y = reference[j].y + idata[i*num_elements+j].y;
	reference[j].z = reference[j].z + idata[i*num_elements+j].z;
	reference[j].w = reference[j].w + idata[i*num_elements+j].w;
      }
    }
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));

    unsigned int mem_size = sizeof( float4) * NUM_ELEMENTS * NUM_ITERS;

    // allocate host memory
    float4* h_idata = (float4*) malloc( mem_size);
    // initalize the memory
    //    for( unsigned int i = 0; i < NUM_ELEMENTS * NUM_ITERS; ++i) 
    for( unsigned int i = 0; i < NUM_ELEMENTS; i++) {
      for ( unsigned int j = 0; j < NUM_ITERS; j++) {
	h_idata[j*NUM_ELEMENTS+i] = make_float4( static_cast<float>(i), static_cast<float>(i), static_cast<float>(i), static_cast<float>(i));
      }
    }

    // allocate device memory
    float4* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );
    

    // allocate device memory for results
    float4* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));
    clock_t * d_timer_vals;
    //unsigned int num_threads = NUM_DMA_THREADS + NUM_ELEMENTS;
    unsigned int num_threads = NUM_ELEMENTS + NUM_ELEMENTS/2;
    unsigned int timer_size = sizeof(clock_t) * 4 * num_threads;
    cutilSafeCall( cudaMalloc( (void**) &d_timer_vals, timer_size));


    // execute the kernel
    cutilCheckError( cutStartTimer( timer));
    dma_4ld<<< 1, num_threads >>>( d_idata, d_odata, d_timer_vals);
    cudaThreadSynchronize();

    // Stop Timer:
    cutilCheckError( cutStopTimer( timer));
    float f_time = cutGetTimerValue( timer);
    cutilCheckError( cutDeleteTimer( timer));
    printf( "Processing time: %f (ms)\n", f_time);
    printf( "Bytes Processed: %d KB\n", static_cast<int>(static_cast<float>(mem_size) / 1024.0 ));

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // allocate mem for the result on host side
    unsigned int out_size = sizeof( float4) * NUM_ELEMENTS;
    float4* h_odata = (float4*) malloc( out_size);
    clock_t* h_timer_vals = (clock_t*) malloc (timer_size);
    
    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, out_size, 
			       cudaMemcpyDeviceToHost) );
    cutilSafeCall( cudaMemcpy( h_timer_vals, d_timer_vals, timer_size, 
			       cudaMemcpyDeviceToHost) );

#ifdef TIMERS_ON
    unsigned int warp0_time = (unsigned int) h_timer_vals[0];
    for (unsigned int i = 0; i < num_threads; i+=32) {
      //unsigned int diffclock = (unsigned int) h_timer_vals[num_threads+i] - (unsigned int) h_timer_vals[i];
      //printf("Warp %d iter cycles = %d\n",i,diffclock);
      printf("Warp %d clock1 = %d\n",i,(unsigned int) h_timer_vals[i] - warp0_time);
      printf("Warp %d clock2 = %d\n",i,(unsigned int) h_timer_vals[num_threads+i] - warp0_time);
      printf("Warp %d clock3 = %d\n",i,(unsigned int) h_timer_vals[2*num_threads+i] - warp0_time);
      printf("Warp %d clock4 = %d\n",i,(unsigned int) h_timer_vals[3*num_threads+i] - warp0_time);
    }
#endif

    // compute reference solution
    float4* reference = (float4*) malloc( out_size );
    computeGoldResults( reference, h_idata, NUM_ELEMENTS, NUM_ITERS);

#ifdef DEBUG_PRINT
    for (unsigned int i = 0; i < NUM_ELEMENTS; ++i) {
      //      if (reference[i].x!=h_odata[i].x) {
      if (i>0) {
	printf("reference[%d].x = %f, h_odata[%d].x = %f, delta from previous element = %f\n",i,reference[i].x,i,h_odata[i].x,reference[i].x-reference[i-1].x);
      } else {
	printf("reference[%d].x = %f, h_odata[%d].x = %f\n",i,reference[i].x,i,h_odata[i].x);
      }
	
	//      }
    }
#endif

    // check result
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        cutilCheckError( cutWriteFilef( "./data/regression.dat",
					(float *)h_odata, 2*NUM_ELEMENTS, 0.0));
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        CUTBoolean res;
        res = cutComparef( (float *)reference, (float *)h_odata, 4*NUM_ELEMENTS);
        printf( "%s\n", (1 == res) ? "PASSED" : "FAILED");
    }

    // Report bandwidths
    float f_bw = static_cast<float>(mem_size) / f_time / static_cast<float>(1000000.0);
    printf( "Bandwidth: %f (GB/s)\n", f_bw);
    

    // cleanup memory
    free( h_idata);
    free( h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();

}

