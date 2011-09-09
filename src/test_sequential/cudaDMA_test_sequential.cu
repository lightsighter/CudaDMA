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

#include "cuda.h"
#include "cuda_runtime.h"

#include "../../include/cudaDMA.h"

#define WARP_SIZE 32

// includes, project

// includes, kernels

#define CUDA_SAFE_CALL(x)					\
	{							\
		cudaError_t err = (x);				\
		if (err != cudaSuccess)				\
		{						\
			printf("Cuda error: %s\n", cudaGetErrorString(err));	\
			exit(1);				\
		}						\
	}

// I hate global variables, but whatever
long total_experiments = 0;

// Forward declaration
template<typename TYPE, int BUFFER_SIZE, int MAX_BYTES,
		int NUM_ITERS, int ALIGNMENT, int ALIGN_OFFSET>
__global__
void dma_4ld( TYPE * g_idata, TYPE * g_odata,
		const int num_elements,
		const int num_compute_threads,
		const int num_dma_threads_per_ld,
		const TYPE zero);


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
// Run a CudaDMASequential experiment
////////////////////////////////////////////////////////////////////////////////
template<typename TYPE>
__host__ void initialize(TYPE &t, float val) { }

template<>
__host__ void initialize(float &t, float val) { t = val; }

template<>
__host__ void initialize(float2 &t, float val) { t.x = val; t.y = val; }

template<>
__host__ void initialize(float4 &t, float val) {t.x = val; t.y = val; t.z = val; t.w = val;}

template<int ALIGN_OFFSET>
__host__ void computeReference(float *reference, float *idata, const unsigned int num_elements, const unsigned int num_iters)
{
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
      reference[i] = 0.0f;
    }
    for( unsigned int i = 0; i < num_iters; i++) 
    {
      for( unsigned int j = 0; j < num_elements; j++) {
	reference[j] = reference[j] + idata[i*num_elements+j+ALIGN_OFFSET];
      }
    }
}

template<int ALIGN_OFFSET>
__host__ void computeReference(float2 *reference, float2 *idata, const unsigned int num_elements, const unsigned int num_iters)
{
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
      reference[i] = make_float2(0.0f, 0.0f);
    }
    for( unsigned int i = 0; i < num_iters; i++) 
    {
      for( unsigned int j = 0; j < num_elements; j++) {
	reference[j].x = reference[j].x + idata[i*num_elements+j+ALIGN_OFFSET].x;
	reference[j].y = reference[j].y + idata[i*num_elements+j+ALIGN_OFFSET].y;
      }
    }
}


template<int ALIGN_OFFSET>
__host__ void computeReference(float4 *reference, float4 *idata, const unsigned int num_elements, const unsigned int num_iters)
{
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
      reference[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for( unsigned int i = 0; i < num_iters; i++) 
    {
      for( unsigned int j = 0; j < num_elements; j++) {
	reference[j].x = reference[j].x + idata[i*num_elements+j+ALIGN_OFFSET].x;
	reference[j].y = reference[j].y + idata[i*num_elements+j+ALIGN_OFFSET].y;
	reference[j].z = reference[j].z + idata[i*num_elements+j+ALIGN_OFFSET].z;
	reference[j].w = reference[j].w + idata[i*num_elements+j+ALIGN_OFFSET].w;
      }
    }
}

__host__
bool compare_results(float *one, float *two, int num_elmts)
{
	for (int i=0; i<num_elmts; i++)
	{
		if (one[i] != two[i])
		{
			printf("Difference at element %d: %f but expected %f",i,one[i],two[i]);
			return false;
		}
	}
	return true;
}

template<typename TYPE>
__host__ TYPE make_zero() { }

template<>
__host__ float make_zero() { return 0.0f; }

template<>
__host__ float2 make_zero() { return make_float2(0.0f,0.0f); }

template<>
__host__ float4 make_zero() { return make_float4(0.0f, 0.0f, 0.0f, 0.0f); }

template<int MAX_BYTES_PER_THREAD, int ALIGNMENT, int ALIGN_OFFSET>
__global__ void __launch_bounds__(1024,1)
dma_ld_corner( float *idata, float *odata, int num_floats,
		int num_compute_threads, int num_dma_threads_per_ld)
{
	extern __shared__ float buffer[];
	
	cudaDMASequential<MAX_BYTES_PER_THREAD, ALIGNMENT>
	  dma0 (1, num_dma_threads_per_ld, num_compute_threads,
		num_compute_threads, num_floats*sizeof(float));

	if (dma0.owns_this_thread())
	{
		float *base_ptr = &(idata[ALIGN_OFFSET]);
		dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
	}
	else
	{
		dma0.start_async_dma();
		dma0.wait_for_dma_finish();
		int iters = num_floats/num_compute_threads;
		int index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			float res = buffer[index+ALIGN_OFFSET];
			odata[index] = res;
			index += num_compute_threads;
		}
		if (index < num_floats)
		{
			float res = buffer[index+ALIGN_OFFSET];
			odata[index] = res;
		}
	}
}

template<int MAX_BYTES_PER_THREAD, int ALIGNMENT, int ALIGN_OFFSET>
__host__ bool run_corner_experiment(int num_floats, int dma_warps)
{
	int shared_buffer_size = (num_floats+ALIGN_OFFSET)*sizeof(float);
	// Check to see if we're using more shared memory than there is, if so return
	if (shared_buffer_size > 49152)
		return true;

	// Allocate the input data
	float *h_idata = (float*)malloc((num_floats+ALIGN_OFFSET)*sizeof(float));
	for (int i=0; i<(num_floats+ALIGN_OFFSET); i++)
		h_idata[i] = float(i);
	
	// Allocate device memory for the input and copy the result down
	float *d_idata;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, (num_floats+ALIGN_OFFSET)*sizeof(float) ));
	CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, (num_floats+ALIGN_OFFSET)*sizeof(float), cudaMemcpyHostToDevice));

	// allocate memory for the result
	float *d_odata;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, num_floats*sizeof(float) ));
	float *h_odata = (float*)malloc(num_floats*sizeof(float));
	for (int i=0; i<num_floats; i++)
		h_odata[i] = 0.0f;
	CUDA_SAFE_CALL( cudaMemcpy( d_odata, h_odata, num_floats*sizeof(float), cudaMemcpyHostToDevice));	

	//int num_compute_warps = (num_floats+WARP_SIZE-1)/WARP_SIZE;
	int num_compute_warps = 1;
	int total_threads = (num_compute_warps + dma_warps)*WARP_SIZE;

	//fprintf(stdout,"Corner case experiment: %ld bytes, %d max bytes per thread, %d DMA warps, %d alignment, %d offset, ",num_floats*sizeof(float), MAX_BYTES_PER_THREAD, dma_warps, ALIGNMENT, ALIGN_OFFSET);
	//fflush(stdout);

	// run the experiment
	dma_ld_corner<MAX_BYTES_PER_THREAD,ALIGNMENT,ALIGN_OFFSET><<<1,total_threads,shared_buffer_size,0>>>(d_idata, d_odata, num_floats, num_compute_warps*WARP_SIZE, dma_warps*WARP_SIZE);	
	CUDA_SAFE_CALL( cudaThreadSynchronize());

	CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, num_floats*sizeof(float), cudaMemcpyDeviceToHost));

	// Check the result
	bool pass = true;
	for (int i=0; i<num_floats; i++)
	{
		if (h_odata[i] != h_idata[i+ALIGN_OFFSET])
		{
			fprintf(stdout,"Corner case experiment: %ld bytes, %d max bytes per thread, %d DMA warps, %d alignment, %d offset, ",num_floats*sizeof(float), MAX_BYTES_PER_THREAD, dma_warps, ALIGNMENT, ALIGN_OFFSET);
			printf("Element %d was expecting %f but received %f, ",i,h_idata[i+ALIGN_OFFSET],h_odata[i]);
			pass = false;
			break;
		}
	}
	if (!pass)
	{
		fprintf(stdout,"Result - %s\n",(pass?"SUCCESS":"FAILURE"));
		fflush(stdout);
	}
	
	CUDA_SAFE_CALL(cudaFree(d_idata));
	CUDA_SAFE_CALL(cudaFree(d_odata));
	free(h_idata);
	free(h_odata);

	total_experiments++;

	return pass;
}

template<int MAX_BYTES_PER_THREAD, int ALIGNMENT, int ALIGN_OFFSET>
__host__ void run_all_corner_experiments(bool &success, int max_dma_warps)
{
	printf("Running all corner experiments for max bytes per thread %d - alignment %d - offset %d...\n",MAX_BYTES_PER_THREAD,ALIGNMENT,ALIGN_OFFSET);
	for (int dma_warps=1; dma_warps <= max_dma_warps; dma_warps++)
	{
		int max_total_floats = MAX_BYTES_PER_THREAD*dma_warps*WARP_SIZE/sizeof(float);
		for (int num_floats=4; num_floats < max_total_floats; num_floats++)
		{
			success = success && run_corner_experiment<MAX_BYTES_PER_THREAD,ALIGNMENT,ALIGN_OFFSET>(num_floats, dma_warps);
		}	
	}
}

template<int ALIGNMENT, int ALIGN_OFFSET>
__host__ void run_all_alignment_tests(bool &success, int max_dma_warps)
{
	{
		const int offset = 0;
		// Handle some strange cases in case max bytes per thread is less than alignment
		if ((4+offset)>=ALIGNMENT) run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		if ((8+offset)>=ALIGNMENT) run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		if ((12+offset)>=ALIGNMENT)run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 32;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 64;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 96;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 128;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 160;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 192;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 224;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 256;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 288;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 320;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 352;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 384;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 416;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 448;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 480;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
	{
		const int offset = 512;
		run_all_corner_experiments<4+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<8+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<12+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<16+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<20+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<24+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<28+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
		run_all_corner_experiments<32+offset,ALIGNMENT,ALIGN_OFFSET>(success,max_dma_warps);
	}
}

template<typename TYPE, int BUFFER_SIZE, int MAX_BYTES,
		int NUM_ITERS, int ALIGNMENT, int ALIGN_OFFSET>
__host__ bool run_experiment(int num_elements, int dma_warps)
{
	unsigned int mem_size = sizeof(TYPE) * (num_elements * NUM_ITERS + ALIGN_OFFSET);		
	unsigned int out_size = sizeof(TYPE) * BUFFER_SIZE;

        TYPE *h_idata = (TYPE*) malloc(mem_size);

	// initialize the memory
	for (unsigned int i = 0; i < num_elements; i++)
	  for (unsigned int j = 0; j < NUM_ITERS; j++)
          {
	    initialize<TYPE>(h_idata[j*num_elements+i+ALIGN_OFFSET], float(i));
	  }

        // allocate device memory
	TYPE *d_idata;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, mem_size));
	CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

	// allocate device memory for the results
	TYPE *d_odata;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, out_size));
	// allocate host memory for checking results
	TYPE *h_odata = (TYPE*) malloc(out_size);
	// Zero out the destination
	for (int i=0; i<BUFFER_SIZE; i++)
		h_odata[i] = make_zero<TYPE>();
	CUDA_SAFE_CALL( cudaMemcpy( d_odata, h_odata, out_size, cudaMemcpyHostToDevice));

	unsigned int num_threads = BUFFER_SIZE + (32 * dma_warps);

	fprintf(stdout,"Experiment: Type-%2ld Elements-(%3d+%2d)=%3d Alignment-%2d Offset-%1d DMA warps-%2d Total Threads-%4d ",sizeof(TYPE), BUFFER_SIZE-32, num_elements-(BUFFER_SIZE-32), num_elements, ALIGNMENT, ALIGN_OFFSET, dma_warps, num_threads); 
	fflush(stdout);

	// run the experiment
	cudaEvent_t start, stop;
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	CUDA_SAFE_CALL(cudaEventRecord(start, 0));
	dma_4ld<TYPE,BUFFER_SIZE,MAX_BYTES,NUM_ITERS,ALIGNMENT,ALIGN_OFFSET><<<1, num_threads,0,0>>>( d_idata, d_odata, num_elements, BUFFER_SIZE, dma_warps*32/2, make_zero<TYPE>()); 
	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	cudaThreadSynchronize();

	// Stop the timer
	CUDA_SAFE_CALL(cudaGetLastError());
	float f_time;
	cudaEventElapsedTime( &f_time, start, stop);

	// copy the results back
	CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, out_size, cudaMemcpyDeviceToHost));
	// compute reference solution
	TYPE *reference = (TYPE*) malloc(out_size);
	computeReference<ALIGN_OFFSET>(reference, h_idata, num_elements, NUM_ITERS);
	bool res = compare_results( (float*)h_odata, (float*)reference, sizeof(TYPE)/sizeof(float)*num_elements);
	
	float f_bw = static_cast<float>(mem_size) / f_time / static_cast<float>(1000000.0);
	fprintf(stdout,"Result-%s Bandwidth-%3.2f GB/s\n",(res?"SUCCESS":"FAILURE"),f_bw);
#if 0
	if (num_elements==64)
	{
		float *out = (float*)h_odata;
		for (int i=0; i<num_elements*sizeof(TYPE)/sizeof(float); i++)
			fprintf(stdout,"%.0f ",out[i]);
		fprintf(stdout,"\n");
		out = (float*)reference;
		for (int i=0; i<num_elements*sizeof(TYPE)/sizeof(float); i++)
			fprintf(stdout,"%.0f ",out[i]);
		fprintf(stdout,"\n");
	}
#endif
	fflush(stdout);

	free(h_idata);
	free(h_odata);
	free(reference);
	CUDA_SAFE_CALL(cudaFree(d_idata));
	CUDA_SAFE_CALL(cudaFree(d_odata));

	// Increment the count of the number of experiments run
	total_experiments++;

	return res;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
__host__
int main( int argc, char** argv) 
{
    //const int min_elements = 32;
    //const int max_elements = 512;
    //const int min_dma_warps = 1;
    //const int max_dma_warps = 8; 
    const int num_iterations = 8192;

    bool success16 = true;
    bool success2_16 = true;
    bool success2_80 = true;
    bool success2_81 = true;
    bool success4_16 = true;
    bool success4_80 = true;
    bool success4_81 = true;
    bool success4_40 = true;
    bool success4_41 = true;
    bool success4_42 = true;
    bool success4_43 = true;

    bool corner16_0 = true;
    bool corner08_0 = true;
    bool corner08_2 = true;
    bool corner04_0 = true;
    bool corner04_1 = true;
    bool corner04_2 = true;
    bool corner04_3 = true;

    CUDA_SAFE_CALL(cudaSetDevice( 0 ));
    // Corner case tests
    {
	run_all_alignment_tests<16,0>(corner16_0,16);
	run_all_alignment_tests<8,0>(corner08_0,16);
	run_all_alignment_tests<8,2>(corner08_2,16);
	run_all_alignment_tests<4,0>(corner04_0,16);
	run_all_alignment_tests<4,1>(corner04_1,16);
	run_all_alignment_tests<4,2>(corner04_2,16);
	run_all_alignment_tests<4,3>(corner04_3,16);	
	//run_corner_experiment<28,16,0>(133,1);
    } 

    // Test the float4 cases
    {
	  for (int elems = 1; elems <= 32; elems++)
	  {
		{
			const int dma_warps = 1;
			success16 = success16 && run_experiment<float4,  64, ( 64*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4,  96, ( 96*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 128, (128*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 160, (160*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 192, (192*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 224, (224*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 256, (256*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 288, (288*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 320, (320*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 352, (352*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 384, (384*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 416, (416*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 448, (448*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 480, (480*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 512, (512*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success16 = success16 && run_experiment<float4,  64, ( 64*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4,  96, ( 96*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 128, (128*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 160, (160*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 192, (192*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 224, (224*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 256, (256*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 288, (288*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 320, (320*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 352, (352*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 384, (384*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 416, (416*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 448, (448*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 480, (480*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 512, (512*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success16 = success16 && run_experiment<float4,  64, /*( 64*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4,  96, ( 96*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 128, /*(128*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 160, /*(160*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 192, (192*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 224, /*(224*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 256, /*(256*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 288, (288*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 320, /*(320*sizeof(float4))/(32*dma_warps)*/64, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 352, /*(352*sizeof(float4))/(32*dma_warps)*/64, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 384, (384*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 416, /*(416*sizeof(float4))/(32*dma_warps)*/80, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 448, /*(448*sizeof(float4))/(32*dma_warps)*/80, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 480, (480*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 512, /*(512*sizeof(float4))/(32*dma_warps)*/96, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success16 = success16 && run_experiment<float4,  64, /*( 64*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4,  96, /*( 96*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 128, (128*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 160, /*(160*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 192, /*(192*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 224, /*(224*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 256, (256*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 288, /*(288*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 320, /*(320*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 352, /*(352*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 384, (384*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 416, /*(416*sizeof(float4))/(32*dma_warps)*/64, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 448, /*(448*sizeof(float4))/(32*dma_warps)*/64, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 480, /*(480*sizeof(float4))/(32*dma_warps)*/64, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 512, (512*sizeof(float4))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success16 = success16 && run_experiment<float4,  64, /*( 64*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4,  96, /*( 96*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 128, /*(128*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 160, (160*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 192, /*(192*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 224, /*(224*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 256, /*(256*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 288, /*(288*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 320, (320*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 352, /*(352*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 384, /*(384*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 416, /*(416*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 448, /*(448*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 480, (480*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 512, /*(512*sizeof(float4))/(32*dma_warps)*/64, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success16 = success16 && run_experiment<float4,  64, /*( 64*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4,  96, /*( 96*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 128, /*(128*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 160, /*(160*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 192, (192*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 224, /*(224*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 256, /*(256*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 288, /*(288*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 320, /*(320*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 352, /*(352*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 384, (384*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 416, /*(416*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 448, /*(448*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 480, /*(480*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 512, /*(512*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success16 = success16 && run_experiment<float4,  64, /*( 64*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4,  96, /*( 96*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 128, /*(128*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 160, /*(160*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 192, /*(192*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 224, (224*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 256, /*(256*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 288, /*(288*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 320, /*(320*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 352, /*(352*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 384, /*(384*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 416, /*(416*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 448, (448*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 480, /*(480*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 512, /*(512*sizeof(float4))/(32*dma_warps)*/48, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success16 = success16 && run_experiment<float4,  64, /*( 64*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4,  96, /*( 96*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 128, /*(128*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 160, /*(160*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 192, /*(192*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 224, /*(224*sizeof(float4))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 256, (256*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 288, /*(288*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 320, /*(320*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 352, /*(352*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 384, /*(384*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 416, /*(416*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 448, /*(448*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 480, /*(480*sizeof(float4))/(32*dma_warps)*/32, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success16 = success16 && run_experiment<float4, 512, (512*sizeof(float4)+32*dma_warps-1)/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
	  }
    }
    // Test the float2 cases
    {
	    for (int elems = 2; elems <= 32; elems+=2)
	    {
		// Load using float4, aligned to 16 bytes
		{
			const int dma_warps = 1;
			success2_16 = success2_16 && run_experiment<float2,  64, ( 64*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2,  96, ( 96*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 128, (128*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 160, (160*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 224, (224*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 288, (288*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 320, (320*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 352, (352*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 416, (416*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 448, (448*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 480, (480*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success2_16 = success2_16 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 128, (128*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/64, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/64, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/64, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success2_16 = success2_16 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success2_16 = success2_16 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success2_16 = success2_16 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 320, (320*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success2_16 = success2_16 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success2_16 = success2_16 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 448, (448*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success2_16 = success2_16 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success2_16 = success2_16 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
	}
	for (int elems = 1; elems <= 32; elems++)
	{
		// Load using float2, aligned to 8 bytes
		{
			const int dma_warps = 1;
			success2_80 = success2_80 && run_experiment<float2,  64, ( 64*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2,  96, ( 96*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 128, (128*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 160, (160*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 224, (224*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 288, (288*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 320, (320*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 352, (352*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 416, (416*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 448, (448*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 480, (480*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success2_80 = success2_80 && run_experiment<float2,  64, ( 64*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 128, (128*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/40, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 320, (320*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/56, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 448, (448*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/64, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success2_80 = success2_80 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2,  96, ( 96*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 288, (288*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/40, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/40, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 480, (480*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success2_80 = success2_80 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 128, (128*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success2_80 = success2_80 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 160, (160*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 320, (320*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 480, (480*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success2_80 = success2_80 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success2_80 = success2_80 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 224, (224*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 448, (448*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success2_80 = success2_80 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success2_80 = success2_80 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}

		// Load using float2, aligned to 8 bytes offset by 8 bytes
		{
			const int dma_warps = 1;
			success2_81 = success2_81 && run_experiment<float2,  64, ( 64*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>( 32+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2,  96, ( 96*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>( 64+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 128, (128*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>( 96+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 160, (160*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(128+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(160+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 224, (224*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(192+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(224+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 288, (288*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(256+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 320, (320*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(288+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 352, (352*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(320+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(352+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 416, (416*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(384+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 448, (448*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(416+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 480, (480*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(448+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success2_81 = success2_81 && run_experiment<float2,  64, ( 64*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>( 32+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>( 64+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 128, (128*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>( 96+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(128+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(160+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 1>(192+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(224+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/40, num_iterations, 8, 1>(256+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 320, (320*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(288+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 8, 1>(320+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(352+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/56, num_iterations, 8, 1>(384+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 448, (448*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(416+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/64, num_iterations, 8, 1>(448+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success2_81 = success2_81 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 32+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2,  96, ( 96*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>( 64+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>( 96+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(128+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(160+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(192+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(224+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 288, (288*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(256+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 1>(288+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 1>(320+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(352+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/40, num_iterations, 8, 1>(384+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/40, num_iterations, 8, 1>(416+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 480, (480*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(448+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/48, num_iterations, 8, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success2_81 = success2_81 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 32+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 64+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 128, (128*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>( 96+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(128+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(160+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(192+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(224+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(256+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(288+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(320+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(352+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 1>(384+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 1>(416+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 1>(448+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success2_81 = success2_81 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 32+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 64+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 96+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 160, (160*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(128+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(160+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(192+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(224+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(256+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 320, (320*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(288+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(320+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(352+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(384+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(416+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 480, (480*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(448+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/32, num_iterations, 8, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success2_81 = success2_81 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 32+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 64+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 96+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>(128+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 192, (192*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(160+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(192+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(224+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(256+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(288+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(320+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 384, (384*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(352+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(384+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(416+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(448+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success2_81 = success2_81 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 32+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 64+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 96+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>(128+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>(160+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 224, (224*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(192+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 256, /*(256*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(224+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(256+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(288+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(320+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(352+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(384+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 448, (448*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(416+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(448+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 512, /*(512*sizeof(float2))/(32*dma_warps)*/24, num_iterations, 8, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success2_81 = success2_81 && run_experiment<float2,  64, /*( 64*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 32+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2,  96, /*( 96*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 64+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 128, /*(128*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>( 96+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 160, /*(160*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>(128+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 192, /*(192*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>(160+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 224, /*(224*sizeof(float2))/(32*dma_warps)*/8, num_iterations, 8, 1>(192+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 256, (256*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(224+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 288, /*(288*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(256+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 320, /*(320*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(288+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 352, /*(352*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(320+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 384, /*(384*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(352+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 416, /*(416*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(384+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 448, /*(448*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(416+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 480, /*(480*sizeof(float2))/(32*dma_warps)*/16, num_iterations, 8, 1>(448+elems, 2*dma_warps);
			success2_81 = success2_81 && run_experiment<float2, 512, (512*sizeof(float2))/(32*dma_warps), num_iterations, 8, 1>(480+elems, 2*dma_warps);
		}
	    }	
    }
    // Test the float cases
    {
	for (int elems = 4; elems <= 32; elems+=4)
	{
		// Load using float4, aligned to 16 bytes
		{
			const int dma_warps = 1;
			success4_16 = success4_16 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/48, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/48, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/48, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/64, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/64, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/64, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success4_16 = success4_16 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success4_16 = success4_16 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/32, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success4_16 = success4_16 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success4_16 = success4_16 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success4_16 = success4_16 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success4_16 = success4_16 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success4_16 = success4_16 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 32+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 64+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>( 96+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(128+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(160+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(192+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(224+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(256+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(288+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(320+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(352+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(384+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(416+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(448+elems, 2*dma_warps);
			success4_16 = success4_16 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 16, 0>(480+elems, 2*dma_warps);
		}

	}	
	for (int elems = 2; elems <= 32; elems+=2)
	{
		// Load using float2, aligned to 8 bytes
		{
			const int dma_warps = 1;
			success4_80 = success4_80 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/32, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/40, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/48, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/56, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/64, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success4_80 = success4_80 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/32, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/32, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/32, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success4_80 = success4_80 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success4_80 = success4_80 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success4_80 = success4_80 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success4_80 = success4_80 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success4_80 = success4_80 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success4_80 = success4_80 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 32+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 64+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>( 96+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(128+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(160+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(192+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(224+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(256+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(288+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(320+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(352+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(384+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(416+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 0>(448+elems, 2*dma_warps);
			success4_80 = success4_80 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 8, 0>(480+elems, 2*dma_warps);
		}

		// Load using float2, aligned to 8 bytes offset by 8 bytes
		{
			const int dma_warps = 1;
			success4_81 = success4_81 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>( 32+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>( 64+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>( 96+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 2>(128+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(160+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/32, num_iterations, 8, 2>(192+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(224+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/40, num_iterations, 8, 2>(256+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(288+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/48, num_iterations, 8, 2>(320+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(352+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/56, num_iterations, 8, 2>(384+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(416+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/64, num_iterations, 8, 2>(448+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success4_81 = success4_81 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 32+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 64+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>( 96+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(128+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(160+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(192+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(224+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 2>(256+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 2>(288+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 2>(320+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(352+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/32, num_iterations, 8, 2>(384+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/32, num_iterations, 8, 2>(416+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/32, num_iterations, 8, 2>(448+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success4_81 = success4_81 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 32+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 64+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 96+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(128+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(160+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(192+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(224+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(256+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(288+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(320+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(352+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 2>(384+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 2>(416+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 2>(448+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/24, num_iterations, 8, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success4_81 = success4_81 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 32+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 64+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 96+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(128+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(160+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(192+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(224+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(256+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(288+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(320+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(352+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(384+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(416+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(448+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success4_81 = success4_81 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 32+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 64+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 96+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(128+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(160+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(192+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(224+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(256+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(288+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(320+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(352+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(384+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(416+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(448+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success4_81 = success4_81 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 32+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 64+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 96+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(128+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(160+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(192+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(224+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(256+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(288+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(320+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(352+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(384+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(416+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(448+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success4_81 = success4_81 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 32+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 64+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 96+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(128+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(160+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(192+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(224+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(256+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(288+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(320+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(352+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(384+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(416+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(448+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 8, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success4_81 = success4_81 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 32+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 64+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>( 96+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(128+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(160+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(192+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(224+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(256+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(288+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(320+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(352+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(384+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(416+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/8, num_iterations, 8, 2>(448+elems, 2*dma_warps);
			success4_81 = success4_81 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 8, 2>(480+elems, 2*dma_warps);
		}

	}
	for (int elems = 1; elems <= 32; elems++)
	{
		// Load using float aligned to 4 bytes
		{
			const int dma_warps = 1;
			success4_40 = success4_40 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>( 32+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float,  96, ( 96*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>( 64+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>( 96+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 160, (160*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(128+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(160+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 224, (224*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(192+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(224+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 288, (288*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(256+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(288+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 352, (352*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(320+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(352+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 416, (416*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(384+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(416+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(448+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success4_40 = success4_40 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>( 32+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>( 64+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>( 96+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(128+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(160+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 0>(192+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(224+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 0>(256+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(288+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/24, num_iterations, 4, 0>(320+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(352+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/28, num_iterations, 4, 0>(384+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(416+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/32, num_iterations, 4, 0>(448+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success4_40 = success4_40 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 32+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float,  96, ( 96*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>( 64+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>( 96+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(128+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(160+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(192+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(224+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 288, (288*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(256+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 0>(288+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 0>(320+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(352+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 0>(384+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 0>(416+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(448+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/24, num_iterations, 4, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success4_40 = success4_40 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 32+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 64+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>( 96+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(128+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(160+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(192+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(224+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(256+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(288+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(320+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(352+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 0>(384+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 0>(416+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 0>(448+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success4_40 = success4_40 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 32+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 64+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 96+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 160, (160*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(128+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(160+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(192+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(224+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(256+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(288+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(320+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(352+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(384+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(416+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(448+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success4_40 = success4_40 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 32+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 64+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 96+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>(128+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(160+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(192+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(224+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(256+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(288+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(320+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(352+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(384+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(416+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(448+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success4_40 = success4_40 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 32+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 64+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 96+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>(128+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>(160+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 224, (224*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(192+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(224+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(256+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(288+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(320+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(352+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(384+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(416+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(448+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 0>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success4_40 = success4_40 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 32+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 64+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>( 96+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>(128+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>(160+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 0>(192+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(224+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(256+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(288+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(320+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(352+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(384+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(416+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 0>(448+elems, 2*dma_warps);
			success4_40 = success4_40 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(480+elems, 2*dma_warps);
		}

		// Load using float aligned to 4 bytes offset by 4 bytes
		{
			const int dma_warps = 1;
			success4_41 = success4_41 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>( 32+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float,  96, ( 96*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>( 64+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>( 96+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 160, (160*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(128+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(160+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 224, (224*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(192+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(224+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 288, (288*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(256+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(288+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 352, (352*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(320+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(352+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 416, (416*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(384+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(416+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(448+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success4_41 = success4_41 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>( 32+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>( 64+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>( 96+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(128+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(160+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 1>(192+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(224+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 1>(256+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(288+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/24, num_iterations, 4, 1>(320+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(352+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/28, num_iterations, 4, 1>(384+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(416+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/32, num_iterations, 4, 1>(448+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success4_41 = success4_41 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 32+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float,  96, ( 96*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>( 64+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>( 96+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(128+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(160+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(192+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(224+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 288, (288*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(256+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 1>(288+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 1>(320+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(352+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 1>(384+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 1>(416+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(448+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/24, num_iterations, 4, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success4_41 = success4_41 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 32+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 64+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>( 96+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(128+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(160+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(192+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(224+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(256+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(288+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(320+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(352+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 1>(384+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 1>(416+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 1>(448+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success4_41 = success4_41 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 32+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 64+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 96+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 160, (160*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(128+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(160+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(192+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(224+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(256+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(288+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(320+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(352+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(384+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(416+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(448+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success4_41 = success4_41 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 32+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 64+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 96+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>(128+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(160+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(192+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(224+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(256+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(288+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(320+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(352+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(384+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(416+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(448+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success4_41 = success4_41 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 32+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 64+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 96+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>(128+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>(160+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 224, (224*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(192+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(224+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(256+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(288+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(320+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(352+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(384+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 0>(416+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(448+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 1>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success4_41 = success4_41 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 32+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 64+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>( 96+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>(128+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>(160+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 1>(192+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(224+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(256+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(288+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(320+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(352+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(384+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(416+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 1>(448+elems, 2*dma_warps);
			success4_41 = success4_41 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 1>(480+elems, 2*dma_warps);
		}



		// Load using float aligned to 4 bytes offset by 8 bytes
		{
			const int dma_warps = 1;
			success4_42 = success4_42 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>( 32+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float,  96, ( 96*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>( 64+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>( 96+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 160, (160*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(128+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(160+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 224, (224*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(192+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(224+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 288, (288*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(256+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(288+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 352, (352*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(320+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(352+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 416, (416*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(384+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(416+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(448+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success4_42 = success4_42 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>( 32+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>( 64+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>( 96+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(128+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(160+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 2>(192+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(224+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 2>(256+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(288+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/24, num_iterations, 4, 2>(320+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(352+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/28, num_iterations, 4, 2>(384+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(416+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/32, num_iterations, 4, 2>(448+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success4_42 = success4_42 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 32+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float,  96, ( 96*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>( 64+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>( 96+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(128+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(160+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(192+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(224+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 288, (288*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(256+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 2>(288+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 2>(320+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(352+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 2>(384+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 2>(416+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(448+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/24, num_iterations, 4, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success4_42 = success4_42 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 32+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 64+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>( 96+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(128+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(160+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(192+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(224+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(256+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(288+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(320+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(352+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 2>(384+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 2>(416+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 2>(448+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success4_42 = success4_42 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 32+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 64+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 96+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 160, (160*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(128+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(160+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(192+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(224+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(256+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(288+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(320+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(352+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(384+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(416+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(448+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success4_42 = success4_42 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 32+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 64+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 96+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>(128+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(160+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(192+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(224+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(256+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(288+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(320+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(352+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(384+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(416+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(448+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success4_42 = success4_42 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 32+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 64+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 96+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>(128+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>(160+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 224, (224*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(192+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(224+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(256+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(288+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(320+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(352+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(384+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(416+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(448+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 2>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success4_42 = success4_42 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 32+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 64+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>( 96+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>(128+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>(160+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 2>(192+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(224+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(256+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(288+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(320+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(352+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(384+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(416+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 2>(448+elems, 2*dma_warps);
			success4_42 = success4_42 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 2>(480+elems, 2*dma_warps);
		}

		// Load using float aligned to 4 bytes offset by 12 bytes
		{
			const int dma_warps = 1;
			success4_43 = success4_43 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>( 32+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float,  96, ( 96*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>( 64+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>( 96+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 160, (160*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(128+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(160+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 224, (224*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(192+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(224+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 288, (288*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(256+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(288+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 352, (352*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(320+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(352+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 416, (416*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(384+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(416+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(448+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 2;
			success4_43 = success4_43 && run_experiment<float,  64, ( 64*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>( 32+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>( 64+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>( 96+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(128+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(160+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 3>(192+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(224+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 3>(256+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(288+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/24, num_iterations, 4, 3>(320+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(352+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/28, num_iterations, 4, 3>(384+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(416+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/32, num_iterations, 4, 3>(448+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 3;
			success4_43 = success4_43 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 32+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float,  96, ( 96*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>( 64+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>( 96+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(128+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(160+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(192+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(224+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 288, (288*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(256+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 3>(288+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 3>(320+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(352+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 3>(384+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/20, num_iterations, 4, 3>(416+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(448+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/24, num_iterations, 4, 3>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 4;
			success4_43 = success4_43 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 32+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 64+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 128, (128*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>( 96+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(128+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(160+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(192+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(224+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(256+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(288+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(320+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(352+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 3>(384+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 3>(416+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 3>(448+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 5;
			success4_43 = success4_43 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 32+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 64+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 96+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 160, (160*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(128+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(160+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(192+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(224+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(256+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 320, (320*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(288+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(320+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(352+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(384+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(416+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 480, (480*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(448+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/16, num_iterations, 4, 3>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 6;
			success4_43 = success4_43 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 32+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 64+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 96+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>(128+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 192, (192*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(160+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(192+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(224+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(256+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(288+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(320+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 384, (384*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(352+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(384+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(416+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(448+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 7;
			success4_43 = success4_43 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 32+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 64+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 96+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>(128+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>(160+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 224, (224*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(192+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 256, /*(256*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(224+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(256+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(288+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(320+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(352+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(384+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 448, (448*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(416+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(448+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 512, /*(512*sizeof(float))/(32*dma_warps)*/12, num_iterations, 4, 3>(480+elems, 2*dma_warps);
		}
		{
			const int dma_warps = 8;
			success4_43 = success4_43 && run_experiment<float,  64, /*( 64*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 32+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float,  96, /*( 96*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 64+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 128, /*(128*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>( 96+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 160, /*(160*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>(128+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 192, /*(192*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>(160+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 224, /*(224*sizeof(float))/(32*dma_warps)*/4, num_iterations, 4, 3>(192+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 256, (256*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(224+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 288, /*(288*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(256+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 320, /*(320*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(288+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 352, /*(352*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(320+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 384, /*(384*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(352+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 416, /*(416*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(384+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 448, /*(448*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(416+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 480, /*(480*sizeof(float))/(32*dma_warps)*/8, num_iterations, 4, 3>(448+elems, 2*dma_warps);
			success4_43 = success4_43 && run_experiment<float, 512, (512*sizeof(float))/(32*dma_warps), num_iterations, 4, 3>(480+elems, 2*dma_warps);
		}
	}
    }

    printf("-------------------------------------------------\n");
    printf("Summary:\n");
    printf("\tFloat4-Alignment16-Offset0: %s\n",(success16?"SUCCESS":"FAILURE"));
    printf("\tFloat2-Alignment16-Offset0: %s\n",(success2_16?"SUCCESS":"FAILURE"));
    printf("\tFloat2-Alignment08-Offset0: %s\n",(success2_80?"SUCCESS":"FAILURE"));
    printf("\tFloat2-Alignment08-Offset1: %s\n",(success2_81?"SUCCESS":"FAILURE"));
    printf("\tFloat -Alignment16-Offset0: %s\n",(success4_16?"SUCCESS":"FAILURE"));
    printf("\tFloat -Alignment08-Offset0: %s\n",(success4_80?"SUCCESS":"FAILURE"));
    printf("\tFloat -Alignment08-Offset2: %s\n",(success4_81?"SUCCESS":"FAILURE"));
    printf("\tFloat -Alignment04-Offset0: %s\n",(success4_40?"SUCCESS":"FAILURE"));
    printf("\tFloat -Alignment04-Offset1: %s\n",(success4_41?"SUCCESS":"FAILURE"));
    printf("\tFloat -Alignment04-Offset2: %s\n",(success4_42?"SUCCESS":"FAILURE"));
    printf("\tFloat -Alignment04-Offset3: %s\n",(success4_43?"SUCCESS":"FAILURE"));
    printf("\n");
    printf("\tCorner Case Alignment16-Offset0: %s\n",(corner16_0?"SUCCESS":"FAILURE"));
    printf("\tCorner Case Alignment08-Offset0: %s\n",(corner08_0?"SUCCESS":"FAILURE"));
    printf("\tCorner Case Alignment08-Offset2: %s\n",(corner08_2?"SUCCESS":"FAILURE"));
    printf("\tCorner Case Alignment04-Offset0: %s\n",(corner04_0?"SUCCESS":"FAILURE"));
    printf("\tCorner Case Alignment04-Offset1: %s\n",(corner04_1?"SUCCESS":"FAILURE"));
    printf("\tCorner Case Alignment04-Offset2: %s\n",(corner04_2?"SUCCESS":"FAILURE"));
    printf("\tCorner Case Alignment04-Offset3: %s\n",(corner04_3?"SUCCESS":"FAILURE"));
    printf("\n\tTotal Experiments - %ld\n",total_experiments);
    return 0;
}


//#define DEBUG_PRINT 1

template<typename TYPE>
__device__ __forceinline__
void update(TYPE &dst, TYPE &src) { dst = src; }

template<>
__device__ __forceinline__
void update(float &dst, float &src) { dst += src; }

template<>
__device__ __forceinline__
void update(float2 &dst, float2 &src) { dst.x += src.x; dst.y += src.y; }

template<>
__device__ __forceinline__
void update(float4 &dst, float4 &src) { dst.x += src.x; dst.y += src.y;
					dst.z += src.z; dst.w += src.w; }

template<typename TYPE, int BUFFER_SIZE, int MAX_BYTES, 
		int NUM_ITERS, int ALIGNMENT, int ALIGN_OFFSET>
__global__ void __launch_bounds__(1024,1)
dma_4ld( TYPE * g_idata, TYPE * g_odata, 
		const int num_elements,
		const int num_compute_threads,
		const int num_dma_threads_per_ld,
		const TYPE zero ) 
{
  // shared memory
  __shared__  TYPE sdata_i0[BUFFER_SIZE + ALIGN_OFFSET];
  __shared__  TYPE sdata_i1[BUFFER_SIZE + ALIGN_OFFSET];

  // Constants

  // access thread id
  unsigned int tid = threadIdx.x ;

  // Preamble
  TYPE acc = zero;

  if (tid<num_compute_threads) {
    sdata_i0[ALIGN_OFFSET+tid] = acc;
    sdata_i1[ALIGN_OFFSET+tid] = acc;
  }
  __syncthreads();

  cudaDMASequential<MAX_BYTES, ALIGNMENT> 
    dma0 (1, num_dma_threads_per_ld, num_compute_threads,
	  num_compute_threads, num_elements*sizeof(TYPE));
  cudaDMASequential<MAX_BYTES, ALIGNMENT>
    dma1 (2, num_dma_threads_per_ld, num_compute_threads,
	  num_compute_threads + num_dma_threads_per_ld, num_elements*sizeof(TYPE));

  if (tid < num_compute_threads) {

    // This is the compute code

    // Pre-amble:
    dma1.start_async_dma();
    TYPE tmp0 = sdata_i0[ALIGN_OFFSET+tid];
    dma0.start_async_dma();
    update(acc, tmp0);
    dma1.wait_for_dma_finish();
    TYPE tmp1 = sdata_i1[ALIGN_OFFSET+tid];
    dma1.start_async_dma();
    update(acc, tmp1);

    for (unsigned int i = 0; i < NUM_ITERS-2; i+=2) {

      // Phase 1:
      dma0.wait_for_dma_finish();
      TYPE tmp0 = sdata_i0[ALIGN_OFFSET+tid];
      dma0.start_async_dma();
      update(acc, tmp0);

      // Phase 2:
      dma1.wait_for_dma_finish();
      TYPE tmp1 = sdata_i1[ALIGN_OFFSET+tid];
      dma1.start_async_dma();
      update(acc, tmp1);

    }

    // Postamble:
    dma0.wait_for_dma_finish();
    TYPE tmp = sdata_i0[ALIGN_OFFSET+tid];
    update(acc, tmp);
    g_odata[tid] = acc;

  } else if (dma0.owns_this_thread()) {

    for (unsigned int j = 0; j < NUM_ITERS; j+=2) {
      TYPE * base_ptr = &g_idata[ALIGN_OFFSET+j*num_elements];
      dma0.execute_dma( base_ptr, &sdata_i0[ALIGN_OFFSET] );
    }

  } else if (dma1.owns_this_thread()) {

    for (unsigned int j = 1; j < NUM_ITERS; j+=2) {
      TYPE * base_ptr = &g_idata[ALIGN_OFFSET+j*num_elements];
      dma1.execute_dma( base_ptr, &sdata_i1[ALIGN_OFFSET] );
    }

  }

}


#if 0
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv, int num_elmts, int num_iters) 
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));

    unsigned int mem_size = sizeof( float4) * num_elmts * num_iters;

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
#endif
