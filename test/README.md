#  Test codes

## 1. cost.cu

Simply use CPU chrono function to calculate the operations in cuda.

### 1.1 basic parameters

First call `print_parameters()` to figure out the parameters in each GPU.

I got the following:

```
compute capability Version: 8.6
Max Threads per block: 1024
Max Threads per Multiprocessor: 1536
```

- **Compute Capability** means the compute capability of this cuda version;
- **Max Threads per Multiprocessor** means the theads num supported by each GPU (SM)

So we cannot run more than 1024 parallely in one block.

### 1.2 sum matrix parallely

The following experiments will set block_num = 1 and thread_per_block = N * N.

|N| cudaMalloc | cudaMemcpy | parallel | serial | cudaFree| 
| :---: | :---: | :---: | :---: | :---: | :---: |
|15|422921|35|8|18|96|
|16|399609|32|9|19|88|
|17|426547|31|8|20|100|
|31|430924|34|10|51|97|
|32|390979|31|11|53|94|
|64|88218|40|2|190|87|

Notice that N=64 is incorrect.

When N\*N larger than 1024 , will get error : status=9, cudaErrorInvalidConfiguration


### 1.3 utilization for scheduling

| Operation | cost(us) |
| :---: | :---: |
| nvmlInit | 7936 |
| nvmlDeviceGetHandleByIndex | 6629 |
| nvmlDeviceGetUtilizationRates | 1681 |
| nvmlShutdown | 3202 |


