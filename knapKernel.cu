#include "knapKernel.h"
#include <stdio.h>
#define    MAX(x,y)   ((x)>(y) ? (x) : (y))
#define    table(i,j)    table[(i)*(C+1)+(j)]

__global__ void FirstRowKernel(long C, long weight, long profit, long* table){
	int table_index = blockDim.x * blockIdx.x + threadIdx.x;

	if(table_index < weight && table_index <= C){
	   table(0, table_index) = 0;
   }else if(table_index >= weight && table_index <= C){
			table(0, table_index) = profit;
   }
}

__global__ void FullTableKernel(long i, long C, long weight, long profit, long* table){
	//will be called with 2 rows of table, uses row i-1 to calculate row i
	int table_index = blockDim.x * blockIdx.x + threadIdx.x;
	
	 if(table_index < weight && table_index <= C){
		 table(i,table_index) = table(i-1,table_index);
	 }else if(table_index >= weight && table_index <= C){
		 table(i,table_index)=MAX(table(i-1,table_index),profit+table(i-1,table_index-weight));
	 }
}