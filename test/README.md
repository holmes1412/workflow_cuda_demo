#  Test codes

## 1. cost.cu

Simply use CPU chrono function to calculate the operations in cuda.

First call `print_parameters()` to figure out the parameters in each GPU.

I got the following:

```
compute capability Version: 8.6
Max Threads per block: 1024
Max Threads per Multiprocessor: 1536
```

- **Compute Capability** means the compute capability of this cuda version;
- **Max Threads per Multiprocessor** means the theads num supported by each GPU (SM)

But it's weird that I can still run more than 1024 or 1536 parallely.

The following experiments will set block_num = 1 and thread_per_block = N * N.

**N=16**

```
cudaMalloc cost = 399609
cudaMemcpy cost = 32
cudaExecute parallel cost = 9
cudaExecute serial cost = 19
check success. c[15][15]=30.099998
cudaFree cost = 88
```

**N=32**

```
cudaMalloc cost = 390979
cudaMemcpy cost = 31
cudaExecute parallel cost = 11
cudaExecute serial cost = 53
check success. c[31][31]=62.099998
cudaFree cost = 94
```

**N=64**
```
cudaMalloc cost = 88218
cudaMemcpy cost = 40
cudaExecute parallel cost = 2
cudaExecute serial cost = 190
check success. c[63][63]=126.100006
cudaFree cost = 87
```

Still get the correct answer when N\*N larger than 1024 or 1536 ?

