
// These are computed from the above parameters
#define DMA_THREADS_PER_CTA ( (SAXPY_KERNEL==saxpy_cudaDMA_doublebuffer) ? 4 : 2 ) * DMA_THREADS_PER_LD
#define THREADS_PER_CTA \
  (SAXPY_KERNEL==saxpy_cudaDMA_doublebuffer) ? (COMPUTE_THREADS_PER_CTA+DMA_THREADS_PER_CTA) : \
  (SAXPY_KERNEL==saxpy_cudaDMA) ? (COMPUTE_THREADS_PER_CTA+DMA_THREADS_PER_CTA) : \
  COMPUTE_THREADS_PER_CTA
