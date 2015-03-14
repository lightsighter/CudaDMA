# CudaDMAHalo #
Please note that the CudaDMAHalo object is still under development.  It is functional, but has not been performance tuned or properly specialized to allow variable number of compile-time constants.

## Pattern Description ##
The CudaDMAHalo object supports data transfer for two dimensional halo regions around rectangular shapes.  The rectangular halo region is characterized by two parameters: `dimx` and `dimy` that specify the number of elements in each of the two dimensions for the rectangle.  The `dimx` parameter corresponds to the dimension that is contiguous in physical memory.  The 'origin' of the problem will correspond to the corner of the rectangular object with the smallest coordinates in both the x and y dimensions.  There are actually two origins in this problem: one in the source memory and one in the destination memory.  They are the same relative to the rectangular object, but are different places in physical memory.

Having defined the basics of the problem, we can now define the parameters that define the CudaDMAHalo object.

  * `ELMT_TYPE` - the base type of each element in the rectangle (any 4, 8, or 16 byte type)
  * `RADIUS` - the number of elements to load around the rectangle on one side (any value between 1 and 8)
  * `CORNERS` - a boolean parameter indicating whether the corners in the halo region should be loaded
  * `ALIGNMENT` - the alignment of the pointers passed to the object.  There are two additional restrictions with `ALIGNMENT` for CudaDMAHalo:
    1. `sizeof(ELMT_TYPE) <= ALIGNMENT`
    1. `(RADIUS*sizeof(ELMT_TYPE))%ALIGNMENT == 0` (i.e. you can use `ALIGNMENT=16` with floats and `RADIUS=4`, but you must use `ALIGNMENT=4` with floats and `RADIUS=5`
  * `dimx` - the size of the rectangular block in the x (contiguous) dimension
  * `dimy` - the size of the rectangular block in the y (non-contiguous) dimension
  * `pitch` - the pitch of the contiguous dimension in the source memory.  We then assume that the pitch will be dense in the destination memory (i.e. `pitch == dimx+2*RADIUS`)

The `execute_dma` function is declared as:

`void execute_dma(void * src_origin, void * dst_origin);`

where the two pointers point to the origin in the source and destination memories, not the location of the initial halo cell.  Note this is slightly different than other CudaDMA objects where `execute_dma` takes pointers to the first element to be transferred.

## Constructors ##
There is currently only a single constructor for CudaDMAHalo as we haven't finished performance tuning yet.  We apply template specialization after performance tuning so in the future you should expect to see multiple CudaDMAHalo constructor options with variable numbers of template arguments.

A model for the current CudaDMAHalo constructor is declared as follows.
```
cudaDMAHalo<ELMT_TYPE, RADIUS, CORNERS, ALIGNMENT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   dimx, dimy, pitch);
```

## Performance Considerations ##
Coming soon once performance tuning is complete.

## Restrictions ##
There is one current restriction to CudaDMAHalo that is an artifact of our current implementation.  We are soliciting user feedback as to whether it is too restrictive, but we believe it only applies in a limited case.  If `sizeof(ELMT_TYPE)==16` and `RADIUS>4` and `ALIGNMENT=4` then the number of DMA warps must be at least 2 and must always be a multiple of 2.