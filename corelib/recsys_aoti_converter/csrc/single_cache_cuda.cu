// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

template<class T>
__global__ void fill_fixed(T* pool,const int* slots,const T* key,const T* val,
    int M,int H,int L,int D,int64_t k0,int64_t k1,int64_t k2,int64_t k3,
    int64_t v0,int64_t v1,int64_t v2,int64_t v3) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=M*H*L*D) return;
  int d=i%D,l=(i/D)%L,h=(i/(D*L))%H,m=i/(D*L*H);
  int64_t base=((int64_t)slots[m]*2*L+l)*H*D+h*D+d;
  pool[base]=key[m*k0+h*k1+l*k2+d*k3];
  pool[base+(int64_t)L*H*D]=val[m*v0+h*v1+l*v2+d*v3];
}

template<class T>
__global__ void gather_fixed(const T* pool,const int* slots,T* key,T* val,
    int B,int H,int L,int D) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=B*H*L*D) return;
  int d=i%D,l=(i/D)%L,h=(i/(D*L))%H,b=i/(D*L*H);
  int64_t base=((int64_t)slots[b]*2*L+l)*H*D+h*D+d;
  key[i]=pool[base]; val[i]=pool[base+(int64_t)L*H*D];
}

void single_fill_cuda(at::Tensor pool,at::Tensor slots,at::Tensor key,at::Tensor value) {
  TORCH_CHECK(pool.is_cuda()&&slots.is_cuda()&&key.is_cuda()&&value.is_cuda(),"CUDA tensors required");
  TORCH_CHECK(pool.is_contiguous()&&slots.is_contiguous()&&slots.scalar_type()==at::kInt,"fixed pool/int32 slots required");
  TORCH_CHECK(key.scalar_type()==pool.scalar_type()&&value.scalar_type()==pool.scalar_type(),"KV/pool dtype mismatch");
  int M=key.size(0),H=pool.size(3),L=pool.size(2),D=pool.size(4);
  TORCH_CHECK(key.sizes()==value.sizes()&&key.dim()==4&&key.size(1)==H&&key.size(2)==L&&key.size(3)==D&&slots.numel()==M,"invalid fill shapes");
  if(!M) return;
  auto stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16,pool.scalar_type(),"single_fill",[&]{
    fill_fixed<scalar_t><<<(M*H*L*D+255)/256,256,0,stream>>>(pool.data_ptr<scalar_t>(),slots.data_ptr<int>(),
      key.data_ptr<scalar_t>(),value.data_ptr<scalar_t>(),M,H,L,D,
      key.stride(0),key.stride(1),key.stride(2),key.stride(3),
      value.stride(0),value.stride(1),value.stride(2),value.stride(3));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor,at::Tensor> single_gather_cuda(at::Tensor pool,at::Tensor slots) {
  TORCH_CHECK(pool.is_cuda()&&slots.is_cuda()&&pool.is_contiguous()&&slots.is_contiguous()&&slots.scalar_type()==at::kInt,"invalid gather tensors");
  int B=slots.numel(),H=pool.size(3),L=pool.size(2),D=pool.size(4);
  auto key=at::empty({B,H,L,D},pool.options());
  auto value=at::empty_like(key);
  if(!B) return {key,value};
  auto stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16,pool.scalar_type(),"single_gather",[&]{
    gather_fixed<scalar_t><<<(B*H*L*D+255)/256,256,0,stream>>>(pool.data_ptr<scalar_t>(),slots.data_ptr<int>(),key.data_ptr<scalar_t>(),value.data_ptr<scalar_t>(),B,H,L,D);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {key,value};
}
