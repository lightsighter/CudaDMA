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
#include "../../include/cudaDMA.h"

#define WARP_SIZE 32

// includes, project

// includes, kernels

#define CUDA_SAFE_CALL(x)					\
	{							\
		cudaError_t err = (x);				\
		if (err != cudaSuccess)				\
		{						\
			fprintf(stdout,"Experiment for element size %ld radius %d corners %d alignment %d align_offset %d dimx %d dimy %d pitch %d dma_warps %d\n",sizeof(ELMT_TYPE), RADIUS, CORNERS, ALIGNMENT, ALIGN_OFFSET, dimx, dimy, pitch, dma_warps); \
			printf("Cuda error: %s\n", cudaGetErrorString(err));	\
			exit(1);				\
		}						\
	}

template<typename TYPE>
__host__ __device__ __forceinline__ void initialize(TYPE &t, float val) { }

template<>
__host__ __device__ __forceinline__ void initialize(float &t, float val) { t = val; }

template<>
__host__ __device__ __forceinline__ void initialize(float2 &t, float val) { t.x = val; t.y = val; }

template<>
__host__ __device__ __forceinline__ void initialize(float4 &t, float val) {t.x = val; t.y = val; t.z = val; t.w = val;}

template<typename TYPE>
__host__ __device__ __forceinline__ float get_val(TYPE &t) { return -1.0f; }

template<>
__host__ __device__ __forceinline__ float get_val(float &t) { return t; }

template<>
__host__ __device__ __forceinline__ float get_val(float2 &t) { return t.x; }

template<>
__host__ __device__ __forceinline__ float get_val(float4 &t) { return t.x; }

template<typename TYPE>
__host__ __forceinline__ bool equal(TYPE one, TYPE two) { return false; }

template<>
__host__ __forceinline__ bool equal(float one, float two) { return (one==two); }

template<>
__host__ __forceinline__ bool equal(float2 one, float2 two) { return (one.x==two.x)&&(one.y==two.y); }

template<>
__host__ __forceinline__ bool equal(float4 one, float4 two) { return (one.x==two.x)&&(one.y==two.y)&&(one.z==two.z)&&(one.w==two.w); }

// I hate global variables, but whatever
long total_experiments = 0;

template<typename ELMT_TYPE, int RADIUS, bool CORNERS>
__host__ bool check_halo(ELMT_TYPE *idata, ELMT_TYPE *odata, int dimx, int dimy, int pitch)
{
	bool pass = true;
	// First check the rows
	{
		// Check the top rows first
		for (int i=0; i<RADIUS; i++)
		{
			// Compute the start of the row
			ELMT_TYPE *src_row_start = idata + (i*pitch) + (CORNERS ? 0 : RADIUS);				
			ELMT_TYPE *dst_row_start = odata + (i*(dimx+2*RADIUS)) + (CORNERS ? 0 : RADIUS);
			for (int j=0; j<(dimx+(CORNERS ? 2*RADIUS : 0)); j++)
			{
				if (!(equal<ELMT_TYPE>(src_row_start[j],dst_row_start[j])))
				{
					fprintf(stdout,"Error on top row %d at element %d\n",i,j);
					pass = false;
					return pass;
				}
			}
		}
		// Check the bottom rows next
		for (int i=0; i<RADIUS; i++)
		{
			// Compute the start of the row
			ELMT_TYPE *src_row_start = idata + (RADIUS+dimy+i)*pitch + (CORNERS ? 0 : RADIUS);
			ELMT_TYPE *dst_row_start = odata + (RADIUS+dimy+i)*(dimx+2*RADIUS) + (CORNERS ? 0 : RADIUS);
			for (int j=0; j<(dimx+(CORNERS ? 2*RADIUS : 0)); j++)
			{
				if (!(equal<ELMT_TYPE>(src_row_start[j],dst_row_start[j])))
				{
					fprintf(stdout,"Error on bottom row %d at element %d\n",i,j);
					pass = false;
					return pass;
				}
			}
		}
	}
	// Then check the side panels
	{
		// Check the left panel
		for (int i=0; i<dimy; i++)
		{
			ELMT_TYPE *src_row_start = idata + (RADIUS+i)*pitch;
			ELMT_TYPE *dst_row_start = odata + (RADIUS+i)*(dimx+2*RADIUS);
			for (int j=0; j<RADIUS; j++)
			{
				if (!(equal<ELMT_TYPE>(src_row_start[j],dst_row_start[j])))
				{
					fprintf(stdout,"Error on left panel at row %d at element %d\n",i,j);
					pass = false;
					return pass;
				}
			}
		}
		// Check the right panel
		for (int i=0; i<dimy; i++)
		{
			ELMT_TYPE *src_row_start = idata + (RADIUS+i)*pitch + RADIUS + dimx;
			ELMT_TYPE *dst_row_start = odata + (RADIUS+i)*(dimx+2*RADIUS) + RADIUS + dimx;
			for (int j=0; j<RADIUS; j++)
			{
				if (!(equal<ELMT_TYPE>(src_row_start[j],dst_row_start[j])))
				{
					fprintf(stdout,"Error on right panel at row %d at element %d\n",i,j);
					pass = false;
					return pass;
				}
			}
		}
	}
	return pass;
}

template<typename ELMT_TYPE>
__host__ void print_2d_array(ELMT_TYPE *data, int rows, int pitch)
{
	for (int i=0; i<rows; i++)
	{
		printf("\t");
		for (int j=0; j<pitch; j++)
		{
			printf("%1.0f ",get_val<ELMT_TYPE>(data[i*pitch+j]));
		}
		printf("\n");
	}	
}

template<int ALIGNMENT, int ALIGN_OFFSET, typename ELMT_TYPE, int RADIUS, bool CORNERS>
__global__ void __launch_bounds__(1024,1)
dma_ld_test ( ELMT_TYPE * idata, ELMT_TYPE * odata, int dimx, int dimy, int pitch, int buffer_size /*number of ELMT_TYPE*/, int num_compute_threads, int num_dma_threads_per_ld)
{
	extern __shared__ float share[];	

	ELMT_TYPE *buffer = (ELMT_TYPE*)share;

	cudaDMAHalo<ELMT_TYPE,RADIUS,CORNERS,ALIGNMENT>
	  dma0 (1, num_dma_threads_per_ld, num_compute_threads,
		num_compute_threads, dimx, dimy, pitch);

	if (dma0.owns_this_thread())
	{
		// Get the pointers to the origin
#ifndef CUDADMA_DEBUG_ON
		ELMT_TYPE *base_ptr = &(idata[ALIGN_OFFSET + RADIUS + RADIUS*pitch]);
		dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET + RADIUS + RADIUS*(dimx+2*RADIUS)]));
#else
		dma0.wait_for_dma_start();
		dma0.finish_async_dma();
#endif
	}
	else
	{
		// Zero out the buffer
		int iters = buffer_size/num_compute_threads;	
		int index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			initialize<ELMT_TYPE>(buffer[index],0.0f);
			index += num_compute_threads;
		}
		if (index < buffer_size)
			initialize<ELMT_TYPE>(buffer[index],0.0f);
		dma0.start_async_dma();
		dma0.wait_for_dma_finish();
		// Now read the buffer out of shared and write the results back
		index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			ELMT_TYPE res = buffer[index];
			odata[index] = res;
			index += num_compute_threads;
		}
		if (index < buffer_size)
		{
			ELMT_TYPE res = buffer[index];
			odata[index] = res;
		}
	}
}	

template<int ALIGNMENT, int ALIGN_OFFSET, typename ELMT_TYPE, int RADIUS, bool CORNERS>
__host__ bool run_experiment(int dimx, int dimy, int pitch,
				int dma_warps)
{
	int shared_buffer_size = ((dimx+2*RADIUS)*(dimy+2*RADIUS) + ALIGN_OFFSET);
	// Check to see if we're using more shared memory than there is, if so return
	if ((shared_buffer_size*sizeof(ELMT_TYPE)) > 49152)
		return true;

	// Allocate the inpute data
	int input_size = (pitch*(dimy+2*RADIUS)+ALIGN_OFFSET);
	ELMT_TYPE *h_idata = (ELMT_TYPE*)malloc(input_size*sizeof(ELMT_TYPE));
	for (int i=0; i<input_size; i++)
		initialize<ELMT_TYPE>(h_idata[i],(rand()%10));
	// Allocate device memory and copy down
	ELMT_TYPE *d_idata;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_idata, input_size*sizeof(ELMT_TYPE)));
	CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, input_size*sizeof(ELMT_TYPE), cudaMemcpyHostToDevice));

	// Allocate the output size
	ELMT_TYPE *h_odata = (ELMT_TYPE*)malloc(shared_buffer_size*sizeof(ELMT_TYPE));
	for (int i=0; i<shared_buffer_size; i++)
		initialize<ELMT_TYPE>(h_odata[i],0.0f);
	// Allocate device memory and copy down
	ELMT_TYPE *d_odata;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_odata, shared_buffer_size*sizeof(ELMT_TYPE)));
	CUDA_SAFE_CALL( cudaMemcpy( d_odata, h_odata, shared_buffer_size*sizeof(ELMT_TYPE), cudaMemcpyHostToDevice));

	int num_compute_warps = 1;
	int total_threads = (num_compute_warps + dma_warps)*WARP_SIZE;

	dma_ld_test<ALIGNMENT,ALIGN_OFFSET,ELMT_TYPE,RADIUS,CORNERS>
		<<<1,total_threads,shared_buffer_size*sizeof(ELMT_TYPE),0>>>
		(d_idata, d_odata, dimx, dimy, pitch, shared_buffer_size,
		 num_compute_warps*WARP_SIZE, dma_warps*WARP_SIZE);
	CUDA_SAFE_CALL( cudaThreadSynchronize());

	CUDA_SAFE_CALL( cudaMemcpy (h_odata, d_odata, shared_buffer_size*sizeof(ELMT_TYPE), cudaMemcpyDeviceToHost));

	// Print out the arrays
	//printf("Source array:\n");
	//print_2d_array<ELMT_TYPE>(&h_idata[ALIGN_OFFSET],dimy+2*RADIUS,pitch);
	//printf("Result array:\n");
	//print_2d_array<ELMT_TYPE>(&h_odata[ALIGN_OFFSET],dimy+2*RADIUS,dimx+2*RADIUS);

	// Check the result
	bool pass = check_halo<ELMT_TYPE, RADIUS, CORNERS>(&h_idata[ALIGN_OFFSET], &h_odata[ALIGN_OFFSET], dimx, dimy, pitch);
	if (!pass)
	{
		fprintf(stdout,"Experiment for element size %ld radius %d corners %d alignment %d align_offset %d dimx %d dimy %d pitch %d dma_warps %d\n",sizeof(ELMT_TYPE), RADIUS, CORNERS, ALIGNMENT, ALIGN_OFFSET, dimx, dimy, pitch, dma_warps);
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

template<int ALIGNMENT, int ALIGN_OFFSET, typename ELMT_TYPE, int RADIUS, bool CORNERS>
__host__
void run_subset_experiments(bool &success, int min_dma_warps, int max_dma_warps, int dma_stride, int max_dimx, int max_dimy)
{
	printf("Running experiments for element size %ld - radius %d - corners %d - alignment %d - offset %d\n",sizeof(ELMT_TYPE), RADIUS, CORNERS, ALIGNMENT, ALIGN_OFFSET);
	for (int dimx = 8; dimx < max_dimx; dimx += (ALIGNMENT/sizeof(ELMT_TYPE) ? (ALIGNMENT/sizeof(ELMT_TYPE)) : 1))
		for (int dimy = 8; dimy < max_dimy; dimy++)
			for (int dma_warps=min_dma_warps; dma_warps<=max_dma_warps; dma_warps+=dma_stride)
			{
				int pitch = 4*dimx;
				success = success && run_experiment<ALIGNMENT,ALIGN_OFFSET,ELMT_TYPE,RADIUS,CORNERS>(dimx,dimy,pitch,dma_warps);
			}
}

template<int ALIGNMENT, int ALIGN_OFFSET>
__host__
bool run_all_experiments(int max_dma_warps, int max_dimx, int max_dimy)
{
	// Note that some experiments can only be run under certain alignments
	bool success = true;
	// float experiments
	if (ALIGNMENT==4) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float,1,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float,1,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*ALIGNMENT/sizeof(float),float,2,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*ALIGNMENT/sizeof(float),float,2,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float,3,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float,3,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*ALIGNMENT/sizeof(float),float,4,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*ALIGNMENT/sizeof(float),float,4,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float,5,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float,5,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*ALIGNMENT/sizeof(float),float,6,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*ALIGNMENT/sizeof(float),float,6,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float,7,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float,7,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*ALIGNMENT/sizeof(float),float,8,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*ALIGNMENT/sizeof(float),float,8,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);

	// float2 experiments
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,1,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,1,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,2,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,2,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,3,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,3,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,4,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,4,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,5,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,5,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,6,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,6,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,7,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	if (ALIGNMENT==4 || ALIGNMENT==8) run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,7,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,8,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET*(ALIGNMENT/sizeof(float2) ? ALIGNMENT/sizeof(float2) : 1),float2,8,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);

	// float4 experiments	
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,1,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,1,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,2,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,2,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,3,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,3,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,4,false>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,4,true>(success,1,max_dma_warps,1,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,5,false>(success,2,max_dma_warps,2,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,5,true>(success,2,max_dma_warps,2,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,6,false>(success,2,max_dma_warps,2,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,6,true>(success,2,max_dma_warps,2,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,7,false>(success,2,max_dma_warps,2,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,7,true>(success,2,max_dma_warps,2,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,8,false>(success,2,max_dma_warps,2,max_dimx,max_dimy);
	run_subset_experiments<ALIGNMENT,ALIGN_OFFSET,float4,8,true>(success,2,max_dma_warps,2,max_dimx,max_dimy);

	return success;
}

__host__
int main()
{
	bool success16_0 = true;
	bool success08_0 = true;
	bool success08_2 = true;
	bool success04_0 = true;
	bool success04_1 = true;
	bool success04_2 = true;
	bool success04_3 = true;

	// An 8192*sizeof(float) element is almost all of shared memory (can only fit one)
	// We probably want to test up to 32 elements per stride if we're going to have up to 16 dma warps
	//success16_0 = success16_0 && run_all_experiments<16,0>(16,64,64);
	//success08_0 = success08_0 && run_all_experiments<8,0>(16,64,64);
	//success08_2 = success08_2 && run_all_experiments<8,1>(16,64,64);
	success04_0 = success04_0 && run_all_experiments<4,0>(16,64,64);
	success04_1 = success04_1 && run_all_experiments<4,1>(16,64,64);
	success04_2 = success04_2 && run_all_experiments<4,2>(16,64,64);
	success04_3 = success04_3 && run_all_experiments<4,3>(16,64,64);
	
	//success16_0 = success16_0 && run_experiment<16,0,float,4,false>(8,8,32,16);

	fprintf(stdout,"\nResults:\n");
	fprintf(stdout,"\tAlignment16-Offset0: %s\n",(success16_0?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment08-Offset0: %s\n",(success08_0?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment08-Offset2: %s\n",(success08_2?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset0: %s\n",(success04_0?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset1: %s\n",(success04_1?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset2: %s\n",(success04_2?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset3: %s\n",(success04_3?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\n\tTotal Experiments - %ld\n", total_experiments);
	return 0;
}

