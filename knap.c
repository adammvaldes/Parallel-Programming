//
// File name   : knap.c
// Author      : DaeGon
// Modified by : Sanjay and Wim 
// Date        : Sep 20
// Description : Dynamic Programming for the 0/1 knapsack
//               problem.  The entire table is saved.
//

#include <stdio.h>
#include <stdlib.h>
#include "timer.h"


#define    MAX(x,y)   ((x)>(y) ? (x) : (y))
#define    table(i,j)    table[(i)*(C+1)+(j)]

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
   
   initialize_timer ();
   start_timer();

   for (j=0; j<=C; j++) {
     if (j<weights[0])
       table(0, j)= 0;
     else
       table(0, j)= profits[0];
   }

     for ( i=1 ; i < N ; i++ ) {     // Enumerating objects 

       
       for ( j=0 ; j <= C ; j++ ) {
         if(j<weights[i])
           table(i,j) = table(i-1,j);  
         else
   	   table(i,j)=MAX(table(i-1,j),profits[i]+table(i-1,j-weights[i]));
       }

     }
   stop_timer();
   time = elapsed_time ();


  
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
	  
	  //NEW
	  /*for(i = 0; i < N; i++){
		  printf("%ld\n",profits[i]);
	  }*/
	  //NEW

   return 0;
}
