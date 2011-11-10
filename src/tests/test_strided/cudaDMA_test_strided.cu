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
#include <assert.h>
#include <math.h>

#include "cuda.h"
#include "cuda_runtime.h"

//#define CUDADMA_DEBUG_ON
#include "cudaDMA.h"

#include "params_directed.h"

#define WARP_SIZE 32

// includes, project

// includes, kernels

#define CUDA_SAFE_CALL(x)					\
	{							\
		cudaError_t err = (x);				\
		if (err != cudaSuccess)				\
		{						\
			printf("Cuda error: %s\n", cudaGetErrorString(err));	\
			exit(false);				\
		}						\
	}

// I hate global variables, but whatever
long total_experiments = 0;

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int NUM_ELMTS, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
dma_ld_test_four ( float *idata, float *odata, int src_stride/*bytes*/, int dst_stride/*bytes*/, int buffer_size /*number of floats*/, int num_compute_threads)
{
	extern __shared__ float buffer[];	

	cudaDMAStrided<ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
	  dma0 (1, num_compute_threads,
		num_compute_threads,
		src_stride,
		dst_stride);

	if (dma0.owns_this_thread())
	{
		float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
		dma0.wait_for_dma_start();
		dma0.finish_async_dma();
#else
		dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
#endif
	}
	else
	{
		// Zero out the buffer
		int iters = buffer_size/num_compute_threads;	
		int index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			buffer[index] = 0.0f;
			index += num_compute_threads;
		}
		if (index < buffer_size)
			buffer[index] = 0.0f;
		dma0.start_async_dma();
		dma0.wait_for_dma_finish();
		// Now read the buffer out of shared and write the results back
		index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			float res = buffer[index];
			odata[index] = res;
			index += num_compute_threads;
		}
		if (index < buffer_size)
		{
			float res = buffer[index];
			odata[index] = res;
		}
	}
}	

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
dma_ld_test_three ( float *idata, float *odata, int src_stride/*bytes*/, int dst_stride/*bytes*/, int buffer_size /*number of floats*/, int num_compute_threads, int num_elmts)
{
	extern __shared__ float buffer[];	

	cudaDMAStrided<ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
	  dma0 (1, num_compute_threads,
		num_compute_threads,
		num_elmts,
		src_stride,
		dst_stride);

	if (dma0.owns_this_thread())
	{
		float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
		dma0.wait_for_dma_start();
		dma0.finish_async_dma();
#else
		dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
#endif
	}
	else
	{
		// Zero out the buffer
		int iters = buffer_size/num_compute_threads;	
		int index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			buffer[index] = 0.0f;
			index += num_compute_threads;
		}
		if (index < buffer_size)
			buffer[index] = 0.0f;
		dma0.start_async_dma();
		dma0.wait_for_dma_finish();
		// Now read the buffer out of shared and write the results back
		index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			float res = buffer[index];
			odata[index] = res;
			index += num_compute_threads;
		}
		if (index < buffer_size)
		{
			float res = buffer[index];
			odata[index] = res;
		}
	}
}	

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT>
__global__ void __launch_bounds__(1024,1)
dma_ld_test_two ( float *idata, float *odata, int src_stride/*bytes*/, int dst_stride/*bytes*/, int buffer_size /*number of floats*/, int num_compute_threads, int num_elmts, int dma_threads)
{
	extern __shared__ float buffer[];	

	cudaDMAStrided<ALIGNMENT,BYTES_PER_ELMT>
	  dma0 (1, dma_threads, num_compute_threads,
		num_compute_threads,
		num_elmts,
		src_stride,
		dst_stride);

	if (dma0.owns_this_thread())
	{
		float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
		dma0.wait_for_dma_start();
		dma0.finish_async_dma();
#else
		dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
#endif
	}
	else
	{
		// Zero out the buffer
		int iters = buffer_size/num_compute_threads;	
		int index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			buffer[index] = 0.0f;
			index += num_compute_threads;
		}
		if (index < buffer_size)
			buffer[index] = 0.0f;
		dma0.start_async_dma();
		dma0.wait_for_dma_finish();
		// Now read the buffer out of shared and write the results back
		index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			float res = buffer[index];
			odata[index] = res;
			index += num_compute_threads;
		}
		if (index < buffer_size)
		{
			float res = buffer[index];
			odata[index] = res;
		}
	}
}	

template<int ALIGNMENT, int ALIGN_OFFSET>
__global__ void __launch_bounds__(1024,1)
dma_ld_test_one ( float *idata, float *odata, int src_stride/*bytes*/, int dst_stride/*bytes*/, int buffer_size /*number of floats*/, int num_compute_threads, int bytes_per_elmt, int num_elmts, int dma_threads)
{
	extern __shared__ float buffer[];	

	cudaDMAStrided<ALIGNMENT>
	  dma0 (1, dma_threads, 
		num_compute_threads,
		num_compute_threads,
		bytes_per_elmt,
		num_elmts,
		src_stride,
		dst_stride);

	if (dma0.owns_this_thread())
	{
		float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
		dma0.wait_for_dma_start();
		dma0.finish_async_dma();
#else
		dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
#endif
	}
	else
	{
		// Zero out the buffer
		int iters = buffer_size/num_compute_threads;	
		int index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			buffer[index] = 0.0f;
			index += num_compute_threads;
		}
		if (index < buffer_size)
			buffer[index] = 0.0f;
		dma0.start_async_dma();
		dma0.wait_for_dma_finish();
		// Now read the buffer out of shared and write the results back
		index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			float res = buffer[index];
			odata[index] = res;
			index += num_compute_threads;
		}
		if (index < buffer_size)
		{
			float res = buffer[index];
			odata[index] = res;
		}
	}
}	

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int NUM_ELMTS, int DMA_THREADS, int NUM_TEMPLATE_PARAMS>
__host__ bool run_experiment(int src_stride /*in floats*/, int dst_stride/*in floats*/)
{
	// check some assertions
	assert(BYTES_PER_ELMT <= src_stride*sizeof(float));
	assert(BYTES_PER_ELMT <= dst_stride*sizeof(float));
	int shared_buffer_size = (NUM_ELMTS*dst_stride + ALIGN_OFFSET);
	// Check to see if we're using more shared memory than there is, if so return
	if ((shared_buffer_size*sizeof(float)) > 49152)
		return true;

	// Allocate the inpute data
	int input_size = (NUM_ELMTS*src_stride+ALIGN_OFFSET);
	float *h_idata = (float*)malloc(input_size*sizeof(float));
	for (int i=0; i<input_size; i++)
		h_idata[i] = float(i);
	// Allocate device memory and copy down
	float *d_idata;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_idata, input_size*sizeof(float)));
	CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, input_size*sizeof(float), cudaMemcpyHostToDevice));

	// Allocate the output size
	int output_size = (NUM_ELMTS*dst_stride+ALIGN_OFFSET);
	float *h_odata = (float*)malloc(output_size*sizeof(float));
	for (int i=0; i<output_size; i++)
		h_odata[i] = 0.0f;
	// Allocate device memory and copy down
	float *d_odata;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_odata, output_size*sizeof(float)));
	CUDA_SAFE_CALL( cudaMemcpy( d_odata, h_odata, output_size*sizeof(float), cudaMemcpyHostToDevice));

	int num_compute_warps = 1;
	int total_threads = (num_compute_warps)*WARP_SIZE + DMA_THREADS;

	switch (NUM_TEMPLATE_PARAMS)
	{
	case 1:
		dma_ld_test_one<ALIGNMENT,ALIGN_OFFSET>
			<<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
			(d_idata, d_odata, src_stride*sizeof(float), dst_stride*sizeof(float), shared_buffer_size, num_compute_warps*WARP_SIZE,
			BYTES_PER_ELMT,NUM_ELMTS,DMA_THREADS);
		break;	
	case 2:
		dma_ld_test_two<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT>
			<<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
			(d_idata, d_odata, src_stride*sizeof(float), dst_stride*sizeof(float), shared_buffer_size, num_compute_warps*WARP_SIZE,
			NUM_ELMTS,DMA_THREADS);
		break;
	case 3:
		dma_ld_test_three<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,DMA_THREADS>
			<<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
			(d_idata, d_odata, src_stride*sizeof(float), dst_stride*sizeof(float), shared_buffer_size, num_compute_warps*WARP_SIZE,
			NUM_ELMTS);
		break;
	case 4:
		dma_ld_test_four<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,DMA_THREADS>
			<<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
			(d_idata, d_odata, src_stride*sizeof(float), dst_stride*sizeof(float), shared_buffer_size, num_compute_warps*WARP_SIZE);
		break;
	default:
		assert(false);
		break;
	}

	CUDA_SAFE_CALL( cudaThreadSynchronize());

	CUDA_SAFE_CALL( cudaMemcpy (h_odata, d_odata, output_size*sizeof(float), cudaMemcpyDeviceToHost));

	// Check the result
	bool pass = true;
	for (int i=0; i<NUM_ELMTS && pass; i++)
	{
		int in_index = ALIGN_OFFSET+i*src_stride;
		int out_index = ALIGN_OFFSET+i*dst_stride;
		for (int j=0; j<(BYTES_PER_ELMT/sizeof(float)); j++)
		{
			//printf("%f ",h_odata[out_index+j]);
			if (h_idata[in_index+j] != h_odata[out_index+j])
			{
				fprintf(stderr,"Experiment: %d element bytes, %d elements, %ld source stride, %ld destination stride, %d DMA warps, %d alignment, %d offset, ",BYTES_PER_ELMT,NUM_ELMTS,src_stride*sizeof(float),dst_stride*sizeof(float),DMA_THREADS/WARP_SIZE,ALIGNMENT,ALIGN_OFFSET);
				fprintf(stderr,"Index %d of element %d was expecting %f but received %f\n", j, i, h_idata[in_index+j], h_odata[out_index+j]);
				pass = false;
				break;
			}
		}
		//printf("\n");
	}
	if (!pass)
	{
		fprintf(stdout,"Result - %s\n",(pass?"SUCCESS":"FAILURE"));
		fflush(stdout);
	}

	// Free up the remaining memory
	CUDA_SAFE_CALL( cudaFree(d_idata));
	CUDA_SAFE_CALL( cudaFree(d_odata));
	free(h_idata);
	free(h_odata);
	
	total_experiments++;
	
	return pass;
}

#if 0
template<int ALIGNMENT, int ALIGN_OFFSET>
__host__
bool run_all_experiments(int max_element_size, int max_element_count,
			int max_dma_warps)
{
	bool pass = true;
	for (int element_size=1; element_size <= max_element_size; element_size++)
	{
	  fprintf(stdout,"Testing cases with element_size %ld - alignment %d - offset %d...\n",element_size*sizeof(float), ALIGNMENT, ALIGN_OFFSET);
	  fflush(stdout);
	  for (int element_count=1; element_count <= max_element_count; element_count++)
	  {
		// Get the initial source stride from the element size with the given alignment
		const int min_stride = element_size + (element_size%(ALIGNMENT/sizeof(float)) ? 
					((ALIGNMENT/sizeof(float))-(element_size%(ALIGNMENT/sizeof(float)))) : 0);
		// Let's only check full stride cases if element_size is divisible by 31 so we can
		// make the search space a little sparser and also test lots of potential strides
		// on weird alignment offsets
		if ((element_size<1024) && (element_size%127)==0)
		{
			// Make each of the strides range from min_stride to 2*min_stride
			// This should cover all of the cases for a given element size
			// Anything larger is modulo equivalent to a smaller stride
			for (int src_stride=min_stride; src_stride <= (2*min_stride); src_stride += (ALIGNMENT/sizeof(float)))
			  for (int dst_stride=min_stride; dst_stride <= (2*min_stride); dst_stride += (ALIGNMENT/sizeof(float)))
			  {
				for (int dma_warps=1; dma_warps <= max_dma_warps; dma_warps++)
				{
					pass = pass && run_experiment<ALIGNMENT,ALIGN_OFFSET>(element_size, 
							element_count,src_stride,dst_stride,dma_warps);
				}
			  }
		}
		else
		{
			// Just test the variable number of dma_warps
			for (int dma_warps=1; dma_warps <= max_dma_warps; dma_warps++)
			{
				pass = pass && run_experiment<ALIGNMENT,ALIGN_OFFSET>(element_size,
						element_count,min_stride,min_stride,dma_warps);
			}
		}
	  }
	}
	return pass;
}
#endif


template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int NUM_ELMTS>
__host__
void run_all_dma_warps(bool &result)
{
	assert(BYTES_PER_ELMT%sizeof(float)==0);
	const int element_size = BYTES_PER_ELMT/sizeof(float);
	const int min_stride = element_size + (element_size%(ALIGNMENT/sizeof(float)) ? 
					((ALIGNMENT/sizeof(float))-(element_size%(ALIGNMENT/sizeof(float)))) : 0);
	const int warp_size=32;
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,1*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,2*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,3*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,4*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,5*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,6*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,7*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,8*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,9*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,10*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,11*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,12*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,13*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,14*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,15*warp_size>(min_stride,min_stride);
	result = result && run_experiment<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,NUM_ELMTS,16*warp_size>(min_stride,min_stride);
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT>
__host__
void run_all_num_elements(bool &result)
{
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,1>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,2>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,3>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,4>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,5>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,6>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,7>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,8>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,9>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,10>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,11>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,12>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,13>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,14>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,15>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,16>(result);
#if 0
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,17>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,18>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,19>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,20>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,21>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,22>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,23>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,24>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,25>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,26>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,27>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,28>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,29>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,30>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,31>(result);
	run_all_dma_warps<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT,32>(result);
#endif
}

#if 0
template<int ALIGNMENT, int ALIGN_OFFSET>
__host__
bool run_all_experiments()
{
	bool result = true;
	{
		const int base=0;
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=64;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=128;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=192;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=256;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=320;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=384;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=448;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=512;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=576;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=640;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=704;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=768;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=832;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=896;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	{
		const int base=960;
#if 0
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+4>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+8>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+12>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+16>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+20>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+24>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+28>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+32>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+36>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+40>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+44>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+48>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+52>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+56>(result);
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+60>(result);
#endif
		run_all_num_elements<ALIGNMENT,ALIGN_OFFSET,base+64>(result);
	}
	
	return result;
}	
#endif

__host__
int main()
{
#if 0
	bool success16_0 = true;
	bool success08_0 = true;
	bool success08_2 = true;
	bool success04_0 = true;
	bool success04_1 = true;
	bool success04_2 = true;
	bool success04_3 = true;
#endif
	// An 8192*sizeof(float) element is almost all of shared memory (can only fit one)
	// We probably want to test up to 32 elements per stride if we're going to have up to 16 dma warps
	//success16_0 = success16_0 && run_all_experiments<16,0>(8192,32,16);
	//success08_0 = success08_0 && run_all_experiments<8,0>(8192,32,16);
	//success08_2 = success08_2 && run_all_experiments<8,2>(8192,32,16);
	//success04_0 = success04_0 && run_all_experiments<4,0>(8192,32,16);
	//success04_1 = success04_1 && run_all_experiments<4,1>(8192,32,16);
	//success04_2 = success04_2 && run_all_experiments<4,2>(8192,32,16);
	//success04_3 = success04_3 && run_all_experiments<4,3>(8192,32,16);

#if 1
	const int element_size = PARAM_ELMT_SIZE/sizeof(float);
	const int min_stride = element_size + (element_size%(PARAM_ALIGNMENT/sizeof(float)) ? 
					((PARAM_ALIGNMENT/sizeof(float))-(element_size%(PARAM_ALIGNMENT/sizeof(float)))) : 0);
	fprintf(stdout,"Experiment: ALIGNMENT-%2d OFFSET-%d ELMT_SIZE-%5d NUM_ELMTS-%2d DMA_WARPS-%2d NUM_TEMPLATES-%d ",PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_NUM_ELMTS,PARAM_DMA_THREADS/32,PARAM_NUM_TEMPLATES); 
	fflush(stdout);
	bool result = run_experiment<PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_NUM_ELMTS,PARAM_DMA_THREADS,PARAM_NUM_TEMPLATES>(min_stride,min_stride);
	fprintf(stdout,"RESULT: %s\n",(result?"SUCCESS":"FAILURE"));
	fflush(stdout);
#else
	bool result = true;
	printf("Running all experiments for ALIGNMENT-%d OFFSET-%d ELMT_SIZE-%d... ",PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE);
	run_all_num_elements<PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE>(result);
	printf("%s\n",(result?"SUCCESS":"FAILURE"));
#endif
	//success16_0 = success16_0 && run_all_experiments<16,0>();

#if 0
	fprintf(stdout,"\nResults:\n");
	fprintf(stdout,"\tAlignment16-Offset0: %s\n",(success16_0?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment08-Offset0: %s\n",(success08_0?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment08-Offset2: %s\n",(success08_2?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset0: %s\n",(success04_0?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset1: %s\n",(success04_1?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset2: %s\n",(success04_2?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset3: %s\n",(success04_3?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\n\tTotal Experiments - %ld\n", total_experiments);
#endif
	return result;
}

