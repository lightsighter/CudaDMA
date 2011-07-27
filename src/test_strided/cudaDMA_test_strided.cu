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
#include <cudaDMA_test_strided_kernel.cu>

int number_of_sizes = 5;
unsigned int el_cnts[] = {64,32,84,128,256};
unsigned int el_szs_in_f4s[] = {4,4,3,2,1};
int number_of_strides = 4;
unsigned int src_strides_in_bytes[] = {4*16, 4*16*2, 4*16*3, 4*16*4};

void
generateSourceData( float4* source_data, int el_cnt, int el_sz_in_f4s, int src_stride)
{
    float val = 1.0;
    for( unsigned int i = 0; i < NUM_ITERS; i++ ) {
        for( unsigned int j = 0; j < el_cnt; j++ ) {
            for( unsigned int k = 0; k < el_sz_in_f4s; k++ ) {
                source_data[k] = make_float4(val, val, val, val);
            }
            source_data += src_stride/sizeof(float4);
        }
    }
}

void
computeGoldResults( float4* reference, int el_cnt, int el_sz_in_f4s)
{
    for( unsigned int i = 0; i < F4S_IN_SHMEM; ++i) {
        reference[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for( unsigned int i = 0; i < el_cnt*el_sz_in_f4s; ++i) {
        for( unsigned int j = 0; j < NUM_ITERS; ++j) {
            reference[i].x += 1.0;
            reference[i].y += 1.0;
            reference[i].z += 1.0;
            reference[i].w += 1.0;
        }
    }
}

void
printDataBlock( float4* data, unsigned int size, unsigned int f4s_per_row)
{
    unsigned int idx = 0;
    while( size > 0) {
        for( unsigned int i = 0; i < f4s_per_row; i++) {
            printf("%1.0f", data[idx++].x);
        }
        printf("\n");
        size -= f4s_per_row*sizeof(float4);
   } 
}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int which_size, int which_stride, int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    for(int i = 0; i < number_of_sizes; i++) 
        for(int j = 0; j < number_of_strides; j++) {
            printf("Testing Stride Pattern [%d elements, %d B/el, %d src stride]:\n",el_cnts[i],el_szs_in_f4s[i]*sizeof(float4), src_strides_in_bytes[j]);
            runTest( i, j, argc, argv);
            printf("\n");
        }
    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int which_size, int which_stride, int argc, char** argv) 
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));

    unsigned int el_cnt = el_cnts[which_size];
    unsigned int el_sz_in_f4s = el_szs_in_f4s[which_size];
    unsigned int src_stride_in_bytes = src_strides_in_bytes[which_stride];

    unsigned int global_mem_sz_in_bytes_per_iter = src_stride_in_bytes*el_cnt;
    unsigned int global_mem_sz_in_bytes = global_mem_sz_in_bytes_per_iter*NUM_ITERS;
    unsigned int output_mem_sz_in_bytes_per_iter = F4S_IN_SHMEM*sizeof(float4);
    unsigned int output_mem_sz_in_bytes = output_mem_sz_in_bytes_per_iter;

    unsigned int num_threads = 2*NUM_DMA_THREADS + F4S_IN_SHMEM;
    unsigned int timer_size = sizeof(clock_t) * 4 * num_threads;

    // allocate host memory
    float4* h_idata = (float4*) calloc( global_mem_sz_in_bytes, sizeof(char));
    float4* h_odata = (float4*) calloc( output_mem_sz_in_bytes, sizeof(char));
    clock_t* h_timer_vals = (clock_t*) malloc (timer_size);

    // allocate device memory
    float4* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, global_mem_sz_in_bytes));
    float4* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, output_mem_sz_in_bytes));
    clock_t * d_timer_vals;
    cutilSafeCall( cudaMalloc( (void**) &d_timer_vals, timer_size));

    // initalize the host memory
    generateSourceData( h_idata, el_cnt, el_sz_in_f4s, src_stride_in_bytes);
    // initialize the device memory
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, global_mem_sz_in_bytes,
                                cudaMemcpyHostToDevice) );
    //printDataBlock( h_idata, global_mem_sz_in_bytes, src_stride_in_bytes/sizeof(float4));


    // execute the kernel
    cutilCheckError( cutStartTimer( timer));
    dma_4ld_strided<<< 1, num_threads >>>( d_idata, d_odata, d_timer_vals, el_cnt, el_sz_in_f4s*sizeof(float4), src_stride_in_bytes);
    cudaThreadSynchronize();
    // Stop Timer:
    cutilCheckError( cutStopTimer( timer));
    float f_time = cutGetTimerValue( timer);
    cutilCheckError( cutDeleteTimer( timer));

    printf( "Processing time: %f (ms)\n", f_time);
    int f_bytes = static_cast<int>(sizeof(float4)*static_cast<float>(el_cnt*el_sz_in_f4s*sizeof(float4)*NUM_ITERS) / 1024.0 );
    printf( "Bytes Processed: %d KB\n", f_bytes); 
    float f_bw = static_cast<float>(f_bytes) / 1024.0 / 1024.0 / f_time * 1e3;
    printf( "Bandwidth: %f (GB/s)\n", f_bw);
    
    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");
    

    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, output_mem_sz_in_bytes, 
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
    float4* reference = (float4*) malloc( output_mem_sz_in_bytes );
    computeGoldResults( reference, el_cnt, el_sz_in_f4s);

#ifdef DEBUG_PRINT 
    for (unsigned int i = 0; i < output_mem_sz_in_bytes/sizeof(float4); ++i) {
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
					(float *)h_odata, output_mem_sz_in_bytes/sizeof(float), 0.0));
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        CUTBoolean res;
        res = cutComparef( (float *)reference, (float *)h_odata, output_mem_sz_in_bytes/sizeof(float));
        printf( "%s\n", (1 == res) ? "PASSED" : "FAILED");
    }

    // cleanup memory
    free( h_idata);
    free( h_odata);
    free(h_timer_vals);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();

}

