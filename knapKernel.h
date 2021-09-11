__global__ void FirstRowKernel(long C, long weight, long profit, long* table);
__global__ void FullTableKernel(long i, long C, long weight, long profit, long* table);