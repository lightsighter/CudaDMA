
/*
 ****************************
 The SGEMV example code below was modified from the Magma BLAS library
 and is governed under the following license:
 ****************************

  -- Innovative Computing Laboratory
  -- Electrical Engineering and Computer Science Department
  -- University of Tennessee
  -- (C) Copyright 2009

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
    notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
    notice,  this list of conditions and the following disclaimer in the
    documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the University of Tennessee, Knoxville nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ****************************

 ****************************
 Some of the example code is also governed under the following license:
 ****************************

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



#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime.h"

#include "../../../include/cudaDMA.h"

#define SIZE_N	        896
#define SIZE_M		SIZE_N

// These defines are only used in the non-DMA version
#define num_threads 128
#define sgemv_bs 32

#ifdef VEC_SINGLE
#define DMA_KERNEL			sgemvn_cuda_dma_vec_single
#define COMPUTE_THREADS_PER_CTA		128	
#define DMA_THREADS_PER_LD		32	
#define DMA_LDS				1
#define VEC_ELMTS		 	128	
#endif

#ifdef VEC_DOUBLE
#define DMA_KERNEL			sgemvn_cuda_dma_vec_double
#define COMPUTE_THREADS_PER_CTA		128	
#define DMA_THREADS_PER_LD		32	
#define DMA_LDS				2
#define VEC_ELMTS			1024
#endif

#ifdef VEC_MANUAL
#define DMA_KERNEL			sgemvn_cuda_dma_vec_manual
#define COMPUTE_THREADS_PER_CTA		128	
#define DMA_THREADS_PER_LD		32
#define DMA_LDS				1
#define VEC_ELMTS			512	
#endif

#ifdef BOTH_SINGLE
#define DMA_KERNEL			sgemvn_cuda_dma_both_single
#define COMPUTE_THREADS_PER_CTA	        128
#define DMA_THREADS_PER_LD		32
#define DMA_LDS				5
#define VEC_ELMTS			32	
#endif

#ifdef BOTH_DOUBLE
#define DMA_KERNEL			sgemvn_cuda_dma_both_double
#define COMPUTE_THREADS_PER_CTA		128	
#define DMA_THREADS_PER_LD		32
#define DMA_LDS				10	
#define VEC_ELMTS			32	
#endif

#ifdef BOTH_MANUAL
#define DMA_KERNEL			sgemvn_cuda_dma_both_manual
#define COMPUTE_THREADS_PER_CTA		64	
#define DMA_THREADS_PER_LD		32
#define DMA_LDS				9	
#define VEC_ELMTS			32	
#endif

#ifdef MAT_SINGLE
#define DMA_KERNEL			sgemvn_cuda_dma_mat_single
#define COMPUTE_THREADS_PER_CTA		128	
#define DMA_THREADS_PER_LD		32
#define DMA_LDS				8	
#define VEC_ELMTS			32	
#endif




#define TOSTRING_(x)	#x
#define TOSTRING(x)	TOSTRING_(x)

#ifdef VEC_SINGLE
__global__ void
sgemvn_cuda_dma_vec_single(int n, int m, int n1, float alpha, float *A, int lda, float *x, float *y)
{
	__shared__ float buff[VEC_ELMTS];

	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_0(1,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);

	if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
	{
		dma_ld_0.start_async_dma();	
		int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA + threadIdx.x;

		A += ind;

		float res = 0.f;

		for(int i=0; i<n1; i += VEC_ELMTS)
		{
			dma_ld_0.wait_for_dma_finish();
			#pragma unroll
			for(int j=0; j < VEC_ELMTS; j++)
			{
				res+=A[0]*buff[j];
				A+=lda;
			}
			dma_ld_0.start_async_dma();
		}

		#if 0
		if (m>n1)
		{
			buff[threadIdx.x]  = x[n1];

			__syncthreads();
			for(int j=0; j<(m-n1); j++)
			{
				 res += A[0]*buff[j];
				 A+=lda;
			}
		  }
		#endif

		if (ind<n)
			y[ind] = alpha * res;
	}
	else if (dma_ld_0.owns_this_thread())
	{
		for (int idx=0; idx<n1; idx += VEC_ELMTS)
		{
			dma_ld_0.execute_dma(x,buff);
			x += VEC_ELMTS;
		}	
		dma_ld_0.wait_for_dma_start();
	}
}
#endif // VEC_SINGLE

#ifdef VEC_DOUBLE 
__global__ void
sgemvn_cuda_dma_vec_double(int n, int m, int n1, float alpha, float *A, int lda, float *x, float *y)
{
	__shared__ float buff0[VEC_ELMTS];
	__shared__ float buff1[VEC_ELMTS];

	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_0(1,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);
	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_1(2,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA+1*DMA_THREADS_PER_LD);

	if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
	{
		dma_ld_0.start_async_dma();	
		dma_ld_1.start_async_dma();
		int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA + threadIdx.x;

		A += ind;

		float res = 0.f;

		for(int i=0; i<n1; i += (VEC_ELMTS*2) )
		{
			dma_ld_0.wait_for_dma_finish();
			#pragma unroll
			for(int j=0; j < VEC_ELMTS; j++)
			{
				res+=A[0]*buff0[j];
				A+=lda;
			}
			dma_ld_0.start_async_dma();

			dma_ld_1.wait_for_dma_finish();
			#pragma unroll
			for (int j=0; j < VEC_ELMTS; j++)
			{
				res+=A[0]*buff1[j];
				A+=lda;
			}
			dma_ld_1.start_async_dma();
		}

		if (ind<n)
			y[ind] = alpha * res;
	}
	else if (dma_ld_0.owns_this_thread())
	{
		for (int i=0; i<n1; i += (2*VEC_ELMTS))
		{
			dma_ld_0.execute_dma(x,buff0);
			x += 2*VEC_ELMTS;
		}	
		dma_ld_0.wait_for_dma_start();
	}
	else if (dma_ld_1.owns_this_thread())
	{
	        x += VEC_ELMTS;
		for (int i=0; i<n1; i += (2*VEC_ELMTS))
		{
			dma_ld_1.execute_dma(x,buff1);
			x += 2*VEC_ELMTS;
		}
		dma_ld_1.wait_for_dma_start();
	}
}
#endif // VEC_DOUBLE

#ifdef VEC_MANUAL
__global__ void
sgemvn_cuda_dma_vec_manual(int n, int m, int n1, float alpha, float *A, int lda, float *x, float *y)
{
	__shared__ float buff0[VEC_ELMTS];
	__shared__ float buff1[VEC_ELMTS];


	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_0(1,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);
	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_1(2,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);

	if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
	{
		dma_ld_0.start_async_dma();	
		dma_ld_1.start_async_dma();
		int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA + threadIdx.x;

		A += ind;

		float res = 0.f;

		for(int i=0; i<n1; i += (VEC_ELMTS*2) )
		{
			dma_ld_0.wait_for_dma_finish();
			#pragma unroll
			for(int j=0; j < VEC_ELMTS; j++)
			{
				res+=A[0]*buff0[j];
				A+=lda;
			}
			dma_ld_0.start_async_dma();
			dma_ld_1.wait_for_dma_finish();
			#pragma unroll
			for (int j=0; j < VEC_ELMTS; j++)
			{
				res+=A[0]*buff1[j];
				A+=lda;
			}
			dma_ld_1.start_async_dma();
		}

		if (ind<n)
			y[ind] = alpha * res;
	}
	else if (dma_ld_0.owns_this_thread())
	{
		for (int i=0; i<n1; i += (VEC_ELMTS*2))
		{
			dma_ld_0.execute_dma(x,buff0);
			x += VEC_ELMTS;
			dma_ld_1.execute_dma(x,buff1);
			x += VEC_ELMTS;
		}	
		dma_ld_0.wait_for_dma_start();
		dma_ld_1.wait_for_dma_start();
	}
}
#endif // VEC_MANUAL

#ifdef BOTH_SINGLE
__global__ void
sgemvn_cuda_dma_both_single(int n, int m, int n1, float alpha, float *A, int lda, float *x, float *y)
{

	__shared__ float buff[VEC_ELMTS];
	__shared__ float mat[VEC_ELMTS][COMPUTE_THREADS_PER_CTA];	

	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_0(1,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);

	cudaDMAStrided<true,16,4*COMPUTE_THREADS_PER_CTA,4*DMA_THREADS_PER_LD,VEC_ELMTS>
	  dma_ld_1(2,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA+1*DMA_THREADS_PER_LD,4*lda);

	if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
	{
		dma_ld_0.start_async_dma();	
		dma_ld_1.start_async_dma();

		float res = 0.f;

		for(int i=0; i<n1; i += VEC_ELMTS)
		{
			dma_ld_0.wait_for_dma_finish();
			dma_ld_1.wait_for_dma_finish();
			#pragma unroll
			for(int j=0; j < VEC_ELMTS; j++)
			{
				res+=mat[j][threadIdx.x]*buff[j];
			}
			dma_ld_0.start_async_dma();
			dma_ld_1.start_async_dma();
		}

		int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA + threadIdx.x;
		if (ind<n)
			y[ind] = alpha * res;
	}
	else if (dma_ld_0.owns_this_thread())
	{
		for (int idx=0; idx<n1; idx += VEC_ELMTS)
		{
			dma_ld_0.execute_dma(x,buff);
			x += VEC_ELMTS;
		}	
		dma_ld_0.wait_for_dma_start();
	}
	else if (dma_ld_1.owns_this_thread())
	{
	  int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA;
	  A += ind;
		for (int idx=0; idx<n1; idx += VEC_ELMTS)
		{
    		        dma_ld_1.execute_dma(A,mat);
			A += (lda*VEC_ELMTS);
		}
		dma_ld_1.wait_for_dma_start();
	}
}
#endif // BOTH_SINGLE

#ifdef BOTH_DOUBLE
__global__ void
sgemvn_cuda_dma_both_double(int n, int m, int n1, float alpha, float *A, int lda, float *x, float *y)
{
	__shared__ float buff0[VEC_ELMTS];
	__shared__ float buff1[VEC_ELMTS];
	__shared__ float mat0[VEC_ELMTS][COMPUTE_THREADS_PER_CTA];	
	__shared__ float mat1[VEC_ELMTS][COMPUTE_THREADS_PER_CTA];

	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_0(1,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);

	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_1(2,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA+1*DMA_THREADS_PER_LD);

	cudaDMAStrided<true,16,4*COMPUTE_THREADS_PER_CTA,4*DMA_THREADS_PER_LD,VEC_ELMTS>
	  dma_ld_2(3,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA+2*DMA_THREADS_PER_LD,4*lda);

	cudaDMAStrided<true,16,4*COMPUTE_THREADS_PER_CTA,4*DMA_THREADS_PER_LD,VEC_ELMTS>
	  dma_ld_3(4,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA+6*DMA_THREADS_PER_LD,4*lda);

	if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
	{
		dma_ld_0.start_async_dma();	
		dma_ld_1.start_async_dma();
		dma_ld_2.start_async_dma();
		dma_ld_3.start_async_dma();

		float res = 0.f;

		for(int i=0; i<n1; i += (VEC_ELMTS*2))
		{
			dma_ld_0.wait_for_dma_finish();
			dma_ld_2.wait_for_dma_finish();
			#pragma unroll
			for(int j=0; j < VEC_ELMTS; j++)
			{
				res+=mat0[j][threadIdx.x]*buff0[j];
			}
			dma_ld_0.start_async_dma();
			dma_ld_2.start_async_dma();

			dma_ld_1.wait_for_dma_finish();
			dma_ld_3.wait_for_dma_finish();
			#pragma unroll
			for (int j=0; j < VEC_ELMTS; j++)
			{
				res+=mat1[j][threadIdx.x]*buff1[j];
			}
			dma_ld_1.start_async_dma();
			dma_ld_3.start_async_dma();
		}

		int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA + threadIdx.x;
		if (ind<n)
			y[ind] = alpha * res;
	}
	else if (dma_ld_0.owns_this_thread())
	{
		for (int idx=0; idx<n1; idx += (VEC_ELMTS*2))
		{
			dma_ld_0.execute_dma(x,buff0);
			x += 2 * VEC_ELMTS;
		}	
		dma_ld_0.wait_for_dma_start();
	}
	else if (dma_ld_1.owns_this_thread())
	{
		x += VEC_ELMTS;
		for (int i=0; i<n1; i += (VEC_ELMTS*2))
		{
			dma_ld_1.execute_dma(x,buff1);
			x += 2 * VEC_ELMTS;
		} 
		dma_ld_1.wait_for_dma_start();
	}
	else if (dma_ld_2.owns_this_thread())
	{
	  int ind = blockIdx.x**COMPUTE_THREADS_PER_CTA;
	  A += ind;
		for (int idx=0; idx<n1; idx += (VEC_ELMTS*2))
		{
			dma_ld_2.execute_dma(A,mat0);
			A += (2*lda*VEC_ELMTS);
		}
		dma_ld_2.wait_for_dma_start();
	}
	else if (dma_ld_3.owns_this_thread())
	{
	  int ind = blockIdx.x**COMPUTE_THREADS_PER_CTA + lda*VEC_ELMTS;
	  A += ind;
		for (int i=0; i<n1; i += (VEC_ELMTS*2))
		{
			dma_ld_3.execute_dma(A,mat1);
			A += (2*lda*VEC_ELMTS);
		}	
		dma_ld_3.wait_for_dma_start();
	}
}
#endif // BOTH_DOUBLE

#ifdef BOTH_MANUAL
__global__ void
sgemvn_cuda_dma_both_manual(int n, int m, int n1, float alpha, float *A, int lda, float *x, float *y)
{
	__shared__ float buff0[VEC_ELMTS];
	__shared__ float buff1[VEC_ELMTS];
	__shared__ float mat0[VEC_ELMTS][COMPUTE_THREADS_PER_CTA];	
	__shared__ float mat1[VEC_ELMTS][COMPUTE_THREADS_PER_CTA];

	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_0(1,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);

	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_1(2,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);

	cudaDMAStrided<true,16,4*COMPUTE_THREADS_PER_CTA,8*DMA_THREADS_PER_LD,VEC_ELMTS>
	  dma_ld_2(3,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA+1*DMA_THREADS_PER_LD,4*lda);

	cudaDMAStrided<true,16,4*COMPUTE_THREADS_PER_CTA,8*DMA_THREADS_PER_LD,VEC_ELMTS>
	  dma_ld_3(4,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA+1*DMA_THREADS_PER_LD,4*lda);

	if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
	{
		dma_ld_0.start_async_dma();	
		dma_ld_1.start_async_dma();
		dma_ld_2.start_async_dma();
		dma_ld_3.start_async_dma();

		float res = 0.f;

		for(int i=0; i<n1; i += (VEC_ELMTS*2))
		{
			dma_ld_0.wait_for_dma_finish();
			dma_ld_2.wait_for_dma_finish();
			#pragma unroll
			for(int j=0; j < VEC_ELMTS; j++)
			{
				res+=mat0[j][threadIdx.x]*buff0[j];
			}
			dma_ld_0.start_async_dma();
			dma_ld_2.start_async_dma();

			dma_ld_1.wait_for_dma_finish();
			dma_ld_3.wait_for_dma_finish();
			#pragma unroll
			for (int j=0; j < VEC_ELMTS; j++)
			{
				res+=mat1[j][threadIdx.x]*buff1[j];
			}
			dma_ld_1.start_async_dma();
			dma_ld_3.start_async_dma();
		}

		int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA + threadIdx.x;
		if (ind<n)
			y[ind] = alpha * res;
	}
	else if (dma_ld_0.owns_this_thread())
	{
		for (int idx=0; idx<n1; idx += (2*VEC_ELMTS))
		{
			dma_ld_0.execute_dma(x,buff0);
			x += VEC_ELMTS;
			dma_ld_1.execute_dma(x,buff1);
			x += VEC_ELMTS;
		}	
		dma_ld_0.wait_for_dma_start();
		dma_ld_1.wait_for_dma_start();
	}
	else if (dma_ld_2.owns_this_thread())
	{
	  int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA;
	  A += ind;
		for (int idx=0; idx<n1; idx += (2*VEC_ELMTS))
		{
			dma_ld_2.execute_dma(A,mat0);
			A += (lda*VEC_ELMTS);
			dma_ld_3.execute_dma(A,mat1);
			A += (lda*VEC_ELMTS);
		}
		dma_ld_2.wait_for_dma_start();
		dma_ld_3.wait_for_dma_start();
	}
}
#endif // BOTH_MANUAL

#ifdef MAT_SINGLE
__global__ void
sgemvn_cuda_dma_mat_single(int n, int m, int n1, float alpha, float *A, int lda, float *x, float *y)
{
	__shared__ float mat[VEC_ELMTS][COMPUTE_THREADS_PER_CTA];	

	cudaDMAStrided<true,16,4*COMPUTE_THREADS_PER_CTA,8*DMA_THREADS_PER_LD,VEC_ELMTS>
	  dma_ld_0(0,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA,4*lda);

	if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
	{
		dma_ld_0.start_async_dma();	

		float res = 0.f;
		int vec_index = 0;

		for(int i=0; i<n1; i += VEC_ELMTS)
		{
			dma_ld_0.wait_for_dma_finish();
			#pragma unroll
			for(int j=0; j < VEC_ELMTS; j++)
			{
				res+=mat[j][threadIdx.x]*x[vec_index++];
			}
			dma_ld_0.start_async_dma();
		}

		int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA + threadIdx.x;
		if (ind<n)
			y[ind] = alpha * res;
	}
	else if (dma_ld_0.owns_this_thread())
	{
	  int ind = blockIdx.x*num_threads;
	  A += ind;
		for (int idx=0; idx<n1; idx += VEC_ELMTS)
		{
			dma_ld_0.execute_dma(A,mat);
			A += (lda*VEC_ELMTS);
		}	
		dma_ld_0.wait_for_dma_start();
	}
}
#endif // MAT_SINGLE

__global__ void 
sgemvn_kernel1_fermi(int n, int m, int n1, float alpha, float* A, int lda, float *x, float *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;

  float res = 0.f;

  for(int i=0; i<n1; i += sgemv_bs ){

    #pragma unroll
    for(int j=0; j < sgemv_bs ; j++){
       res += A[0] * x[j];
       A   += lda;
    }
	x += sgemv_bs;
  }

#if 0
  if (m>n1){

     for(int j=0; j<(m-n1); j++){
         res += A[0] * x[j];
         A   += lda;
     }
  }
#endif

  if (ind<n)
     y[ind] = alpha * res;

}


__global__ void 
sgemvn_kernel2_fermi(int n, int m, int n1, float alpha,  float* A, int lda, float *x, float *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;
  x += threadIdx.x;

  float res = 0.f;

  __shared__ float buff[num_threads];
  for(int i=0; i<n1; i += num_threads ){
    __syncthreads();
    buff[threadIdx.x]  = x[i];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < num_threads ; j++){
       res+=A[0]*buff[j];
       A+=lda;
    }
  }
#if 0
  __syncthreads();

  if (m>n1){
     buff[threadIdx.x]  = x[n1];

     __syncthreads();
     for(int j=0; j<(m-n1); j++){
         res += A[0]*buff[j];
         A+=lda;
     }
  }
#endif

  if (ind<n)
     y[ind] = alpha * res;
}




__host__
float run_experiment( void (*kernel)(int,int,int,float,float*,int,float*,float*), int nreps, dim3 grid, dim3 threads, int n, int m, int n1, float alpha, float *mat_d, int lda, float *x_d, float *y_d, float *mat_h, float *x_h, float *y_h)
{
	printf("Running experiment with grid (%d,%d)x(%d,%d)\n",grid.x,grid.y,threads.x,threads.y);	

	// Copy the necessary memory down to the device
	cudaMemcpy(mat_d,mat_h,m*n*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(x_d,x_h,m*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(y_d,y_h,n*sizeof(float),cudaMemcpyHostToDevice);

	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)
	{
		kernel<<<grid,threads,0>>>(n,m,n1,alpha,mat_d,lda,x_d,y_d);
	}
	cudaEventRecord( stop, 0 );

	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("%s\n", cudaGetErrorString(err));
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	// Copy the result back from the device
	cudaMemcpy(y_h,y_d,n*sizeof(float),cudaMemcpyDeviceToHost);

	return elapsed_time_ms;
}

__host__
void initialize_host(float *matA, float *vecX, float *vecY_ref, float *vecY_dma, int m, int n)
{
	for (int i=0; i<n; i++)
		for (int j=0; j<m; j++)
			matA[j*n+i] = float((i*j)%11);

	for (int i=0; i<m; i++)
		vecX[i] = float(i%11);

	for (int i=0; i<n; i++)
	{
		vecY_ref[i] = 0.0f;
		vecY_dma[i] = 0.0f;
	}
}

__host__
void verify(float alpha, float *matA, float *x, float *y, int m, int n, float eps)
{
	for (int i=0; i<n; i++)
	{
		float res = 0.0f;
		for (int j=0; j<m; j++)
		{
			res += (matA[j*n+i] * x[j]); 
		}	
		res *= alpha;
		if (fabs(res - y[i]) > eps)
		{
			printf("Error at row %d!\n",i);
			printf("Expected %.0f but received %.0f\n",res,y[i]);	
			return;
		}
	}
	printf("Success!\n");
}

__host__
int main()
{
	int device = 0;
        cudaSetDevice(device);
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        printf("running on: %s\n", properties.name);
        printf("memory: %.4f GB\n", properties.totalGlobalMem/(1024.f*1024.f*1024.f) );	

	int n = SIZE_N; 
	int m = SIZE_M;
	float alpha = 2.0f;

	float eps_diff = 0.1f;
	int nreps = 100;

	printf("Running experiment with M=%d N=%d\n",m,n);
	
	int nbytes = m*n*sizeof(float) + m*sizeof(float) + n*sizeof(float);	

	printf("Allocating %.2f MB on the device\n",float(nbytes)/(1024.f*1024.f));

	float *matA_h=0, *vecX_h=0, *vecY_h_ref=0, *vecY_h_dma=0;
	float *matA_d=0, *vecX_d=0, *vecY_d=0;

	// Malloc host memory
	matA_h = (float*)malloc(m*n*sizeof(float));	
	vecX_h = (float*)malloc(m*sizeof(float));
	vecY_h_ref = (float*)malloc(n*sizeof(float));
	vecY_h_dma = (float*)malloc(n*sizeof(float));

	// Initialize the host memory
	initialize_host(matA_h,vecX_h,vecY_h_ref,vecY_h_dma,m,n);

	// Allocate the device memory
	cudaMalloc((void**)&matA_d,m*n*sizeof(float));
	cudaMalloc((void**)&vecX_d,m*sizeof(float));
	cudaMalloc((void**)&vecY_d,n*sizeof(float));

	// Run the experiments;
	float ref_time=0.0f, dma_time=0.0f;
	{
	        int n1; 
		/*
		if(n<=8500) 
		  //if (0)
		  n1 = (m/sgemv_bs)*sgemv_bs;
		else
		*/
		  n1 = (m/num_threads)*num_threads;
		  
		int blocks;
		if (n % num_threads==0)
			blocks = n/num_threads;
		else
			blocks = n/num_threads + 1;

		dim3 grid(blocks, 1, 1);
		dim3 threads(num_threads, 1, 1);

		/*
		if(n<=8500) 
		  //if (0)
		  ref_time = run_experiment(sgemvn_kernel1_fermi,nreps,grid,threads,n,m,n1,alpha,matA_d,n,vecX_d,vecY_d,matA_h,vecX_h,vecY_h_ref);
		else
		*/
		  ref_time = run_experiment(sgemvn_kernel2_fermi,nreps,grid,threads,n,m,n1,alpha,matA_d,n,vecX_d,vecY_d,matA_h,vecX_h,vecY_h_ref);


	}
	{
		int n1 = (m/COMPUTE_THREADS_PER_CTA)*COMPUTE_THREADS_PER_CTA;
		int blocks;
		if (n % COMPUTE_THREADS_PER_CTA==0)
			blocks = n/COMPUTE_THREADS_PER_CTA;
		else
			blocks = n/COMPUTE_THREADS_PER_CTA + 1;

		dim3 grid(blocks, 1, 1);
		dim3 threads(COMPUTE_THREADS_PER_CTA+DMA_LDS*DMA_THREADS_PER_LD, 1, 1);
		dma_time = run_experiment(DMA_KERNEL,nreps,grid,threads,n,m,n1,alpha,matA_d,n,vecX_d,vecY_d,matA_h,vecX_h,vecY_h_dma);
	}

	printf("\n");
	float ref_gflops = float(2*m*n)/(ref_time*1e6f);
	float ref_gbs = float((m*n+m+n)*sizeof(float))/(ref_time*1e6f);
	float dma_gflops = float(2*m*n)/(dma_time*1e6f);
	float dma_gbs = float((m*n+m+n)*sizeof(float))/(dma_time*1e6f);
	printf("%50s","magma_sgemvn");
	printf("%9.2f ms %9.2f GFLOPS %9.2f GB/s\n",ref_time,ref_gflops,ref_gbs);
	printf("%50s",TOSTRING(DMA_KERNEL));
	printf("%9.2f ms %9.2f GFLOPS %9.2f GB/s\n",dma_time,dma_gflops,dma_gbs);

	printf("..........................................\n");
	// Verify correctness
	printf("Verifying reference... "); 
	verify(alpha, matA_h, vecX_h, vecY_h_ref,m,n,eps_diff);
	printf("Verifying cudaDMA... ");
	verify(alpha, matA_h, vecX_h, vecY_h_dma,m,n,eps_diff);
	
	// Clean up memory
	cudaFree(matA_d);
	cudaFree(vecX_d);
	cudaFree(vecY_d);
	
	free(matA_h);
	free(vecX_h);
	free(vecY_h_ref);
	free(vecY_h_dma);

	return 0;
}
