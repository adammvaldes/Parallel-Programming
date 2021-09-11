#include <stdio.h>
#include <stdlib.h>
#include "timercu.h"


#define    MAX(x,y)   ((x)>(y) ? (x) : (y))
#define    table(i,j)    table[(i)*(C+1)+(j)]
//CUDA_CHECK_RETURN is from vecMax.cu from PA4 of CS475
#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error: %s at line %d in file %s\n",          \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

#include "knapKernel.h"

void print_table(long* table, long N, long C){
	   printf("j= : ");
   for(int j=0;j<=C;j++){
	   printf("%d ",j);
   }
   printf("\n");
   for(int i=0;i<N;i++){
	   printf("i=%d: ",i);
	   for(int j=0;j<=C;j++){
		   printf("%d ",table(i,j));
	   }
	   printf("\n");
   }
}

int main(int argc, char **argv) {

   FILE   *fp;
   long    N, C;                   // # of objects, capacity 
   long    *weights, *profits;     // weights and profits
   int    verbose,count;

   // Temp variables
   long    i, j, size;

   // Time
   double time;

   // Read input file (# of objects, capacity, (per line) weight and profit )
   if ( argc > 1 ) {
      fp = fopen(argv[1], "r"); 
      if ( fp == NULL) {
         printf("[ERROR] : Failed to read file named '%s'.\n", argv[1]);
         exit(1);
      }
   } else {
      printf("USAGE : %s [filename].\n", argv[0]);
      exit(1);
   }

   if (argc > 2) verbose = atoi(argv[2]); else verbose = 0;

   fscanf(fp, "%ld %ld", &N, &C);
   printf("The number of objects is %d, and the capacity is %d.\n", N, C);

   size    = N * sizeof(long);
   weights = (long *)malloc(size);
   profits = (long *)malloc(size);

   if ( weights == NULL || profits == NULL ) {
      printf("[ERROR] : Failed to allocate memory for weights/profits.\n");
      exit(1);
   }

   for ( i=0 ; i < N ; i++ ) {
      count = fscanf(fp, "%ld %ld", &(weights[i]), &(profits[i]));
      if ( count != 2 ) {
         printf("[ERROR] : Input file is not well formatted.\n");
         exit(1);
      }
   }

   fclose(fp);

   // Solve for the optimal profit
   size = (C+1) * sizeof(long);

      long *table;
      size  = (C+1) * N * sizeof(long);
      table = (long *)malloc(size);
      if ( table == NULL ) {
         printf("[ERROR] : Failed to allocate memory for the whole table.\n");
         exit(1);
      }
   
   
   
   CUDA_CHECK_RETURN(cudaSetDevice(0));
   cudaDeviceSynchronize();
   long* GPU_row;
   long size1, size2;//, size3;
   size1 = (C+1) * sizeof(long);
   CUDA_CHECK_RETURN(cudaMalloc((void**)&GPU_row, size1));
   cudaMemcpy(GPU_row, table, size1, cudaMemcpyHostToDevice);
   int threads_per_block = 32;
   int num_blocks = (C/threads_per_block)+1;
   
   
   initialize_timer ();
   start_timer();
   
   //build the first row
   FirstRowKernel<<<num_blocks, threads_per_block>>>(C, weights[0], profits[0], GPU_row);
   cudaDeviceSynchronize();
   cudaMemcpy(table,GPU_row,size1,cudaMemcpyDeviceToHost);
   cudaFree(GPU_row);
   
   
   //conditional compilation: either use the full table in each kernel call or do it in separate chunks of rows
   #ifdef FULL
		long* GPU_table;
	   size2 = size;
	   CUDA_CHECK_RETURN(cudaMalloc((void**)&GPU_table, size2));
	   for(i=1; i<N; i++){
		   cudaMemcpy(GPU_table, table, size, cudaMemcpyHostToDevice);
		   FullTableKernel<<<num_blocks, threads_per_block>>>(i, C, weights[i], profits[i], GPU_table);
		   cudaDeviceSynchronize();
		   cudaMemcpy(table,GPU_table,size,cudaMemcpyDeviceToHost);
	   }
	   cudaFree(GPU_table);
   #else
	   //cant allocate the full table with cudaMalloc for the larger input files (k100x20M.txt)
	   //so only allocate 2 rows at a time: the row used for the calculation, and the row to be calculated (i and i-1)
	   //downside: additional overhead
	   long* GPU_2rows;
	   size2 = (C+1) * sizeof(long) * 2;
	   CUDA_CHECK_RETURN(cudaMalloc((void**)&GPU_2rows, size2));
	   long* h_2rows = &table(0,0);
	   for(i=1; i<N; i++){
		   cudaMemcpy(GPU_2rows, h_2rows, size2, cudaMemcpyHostToDevice);
		   FullTableKernel<<<num_blocks, threads_per_block>>>(1, C, weights[i], profits[i], GPU_2rows);
		   cudaDeviceSynchronize();
		   cudaMemcpy(&table(i-1,0),GPU_2rows,size2,cudaMemcpyDeviceToHost);
		   h_2rows = &table(i,0);
	   }
	   cudaFree(GPU_2rows);
   #endif
   
   
   stop_timer();
   time = elapsed_time ();
   printf("Time: %lf\n",time);


  
   // End of "Solve for the optimal profit"

   // Backtracking
      int c=C;
      int solution[N];

      for ( i=N-1 ; i > 0 ; i-- ) {
         if ( c-weights[i] < 0 ) {
	   //printf("i=%d: 0 \n",i);
            solution[i] = 0;
         } else {
            if ( table(i-1,c) > table(i-1,c-weights[i]) + profits[i] ) {

	      //printf("i=%d: 0 \n",i);
               solution[i] = 0;
            } else {
	      //printf("i=%d: 1 \n",i);
               solution[i] = 1;
               c = c - weights[i];
            }
         }
      } 
      //wim: first row does not look back
      if(c<weights[0]){
        //printf("i=0: 1 \n");
	solution[0]=0;
      } else {
        //printf("i=0: 0 \n");
        solution[0]=1;
      }

      printf("The optimal profit is %d \nTime taken : %lf.\n", table(N-1,C), time);
     

      if (verbose==1) {

      printf("Solution vector is: ");
      for (i=0 ; i<N ; i++ ) {
         printf("%d ", solution[i]);
      }
      printf("\n");
      }

      if (verbose==2) {
	for (j=1; j<=C; j++){
	  printf ("%d\t", j);
	  for (i=0; i<N; i++)
	    printf ("%d ", table(i, j));
	  printf("\n");
	}
      }
	  //cudaFree(GPU_2rows);
	  //NEW
	  /*for(i = 0; i < N; i++){
		  printf("%ld\n",profits[i]);
	  }*/
	  //NEW

   return 0;
}
