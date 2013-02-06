#
#  Copyright 2013 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

all: stencil_k20

# Important performance note: always compile CudaDMA with abi=no passed to ptxas
# This guarantees that all buffers are stored in registers and all functions are inlined

PTXAS_OPTIONS="-abi=no -v"

# WARNING: this version exercises an nvcc correctness bug!
stencil:
	nvcc -o stencil2D -O2 -arch=compute_20 -code=sm_20 -Xptxas $(PTXAS_OPTIONS) cudaDMA_stencil2D.cu

stencil_k20:
	nvcc -o stencil2D -O2 -arch=compute_35 -code=sm_35 -Xptxas $(PTXAS_OPTIONS) cudaDMA_stencil2D.cu

clean:
	rm -f *.o stencil2D
