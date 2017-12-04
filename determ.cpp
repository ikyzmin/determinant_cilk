#include <stdio.h>
#include <stdlib.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <cilk/cilk_api.h>
#include <iostream>
#include <omp.h>
#include <math.h>
#include <cstdint>
#include <cstring>

void printVec(bool *A,int size){
	for (int i=0;i<size;i++){
		  printf("%d ",A[i]);		
	}
}

/*
Вычисление определителя последовательным кодом
A - развернутая в вектор матрица
disabledCols - вычеркнутые столбцы
row - предыдущая строка 
size - размер квадратной матрицы (size x size)
curSliceSize - текущий размер матрицы 
*/
double serialDet(int * A,bool *disabledCols,int row,int size, int curSliceSize){
	double det = 0;
	int delta = 0;
	
	
	if (curSliceSize == 1) {
		int column = 0; 
		for (int i=0;i<size;i++){
			column = !disabledCols[i] ? i : column;
		}
		det =  A[(row)*size+column];
		return det;
	}
	
	for (int i=0;i<size;i++){
		if (!disabledCols[i]){	
			int curCol = i-delta;
			disabledCols[i] = true;
			det = det + (pow(-1.0,curCol))*A[row*size+i]*serialDet(A,disabledCols,row+1,size,curSliceSize-1);
			disabledCols[i] =  false;
		}else{
			delta++;		
		}
	}
	return det;
}

double serialDetOmp(int * A,bool *disabledCols,int row,int size, int curSliceSize){
	double det_serial = 0;
	double det = 0;
	int delta = 0;
	
	if (curSliceSize == 1) {
		int column = 0; 
		for (int i=0;i<size;i++){
			column = !disabledCols[i] ? i : column;
		}
		det_serial =  A[(row)*size+column];
		return det_serial;
	}
	int i=0;
#pragma omp parallel for shared(A,size,curSliceSize) private (i) firstprivate(disabledCols,delta,row) reduction(+:det)
	for (i=0;i<size;i++){
		bool *disabledColsCopy;
			disabledColsCopy = (bool*)calloc(size,sizeof(bool));
			memcpy(disabledColsCopy, disabledCols, size*sizeof(bool));
		if (!disabledColsCopy[i]){	
			int curCol = i-delta;
			disabledColsCopy[i] = true;
			det += (pow(-1.0,curCol))*A[row*size+i]*serialDet(A,disabledColsCopy,row+1,size,curSliceSize-1);
			disabledColsCopy[i] =  false;
		}else{
			delta++;		
		}
	}
	return det;
}

double serialDetCilk(int * A,bool *disabledCols,int row,int size, int curSliceSize){
	int delta = 0;
	double det_serial = 0;
	cilk::reducer<cilk::op_add<double> > det(0);
	
	if (curSliceSize == 1) {
		int column = 0; 
		for (int i=0;i<size;i++){
			column = !disabledCols[i] ? i : column;
		}
		det_serial =  A[(row)*size+column];
		return det_serial;
	}

	cilk_for(int i=0;i<size;i++){
			bool *disabledColsCopy;
			disabledColsCopy = (bool*)calloc(size,sizeof(bool));
			memcpy(disabledColsCopy, disabledCols, size*sizeof(bool));
		if (!disabledColsCopy[i]){	
			int curCol = i-delta;
			disabledColsCopy[i] = true;
			*det += (pow(-1.0,curCol))*A[row*size+i]*serialDet(A,disabledColsCopy,row+1,size,curSliceSize-1);
			disabledColsCopy[i] =  false;
		}else{
			delta++;		
		}
	}
	return det.get_value();
}


void printMatr(int *A,int size){
	for (int i=0;i<size;i++){
		for (int j=0;j<size;j++){
		  	printf("%d ",A[i*size+j]);	
		}	
		printf("\n");
	}
}

void createMockArray(int *A, int size){
	for (int i=0;i<size;i++){
		for (int j=0;j<size;j++){
		  	A[i*size+j] = rand()%10 +1;	
		}	
		
	}
}

void createDummyMockArray(int *A){
	A[0] = 1;
	A[1] = 2;
	A[2] = 3;
	A[3] = 1;
	A[4] = 42;
	A[5] = 3;
	A[6] = 1;
	A[7] = 12;
	A[8] = 6;

}

int main(){
	int lengths[] = {2,4,8,10,12};
	int n= 3;
	int * A;
	bool *disabledCols;
	omp_set_num_threads(4); 
	__cilkrts_set_param("nworkers","4");
	srand(time(NULL));
	for (int i=0;i<5;i++){
		A = (int*)calloc(lengths[i]*lengths[i],sizeof(int));
		disabledCols = (bool*)calloc(lengths[i],sizeof(bool));
		createMockArray(A,lengths[i]);
		printf("A(%dx%d)----------------------\n",lengths[i],lengths[i]);
		double start = omp_get_wtime();
		printf("det = %f\n",serialDet(A,disabledCols,0,lengths[i],lengths[i]));
		double finish = omp_get_wtime();
		printf("time = %f\n",finish-start);
		free(A);
		free(disabledCols);	
	}
for (int i=0;i<5;i++){
		A = (int*)calloc(lengths[i]*lengths[i],sizeof(int));
		disabledCols = (bool*)calloc(lengths[i],sizeof(bool));
		createMockArray(A,lengths[i]);
		printf("A(%dx%d)----------------------\n",lengths[i],lengths[i]);
		double start = omp_get_wtime();
		printf("Omp det = %f\n",serialDetOmp(A,disabledCols,0,lengths[i],lengths[i]));
		double finish = omp_get_wtime();
		printf("time = %f\n",finish-start);
		free(A);
		free(disabledCols);	
	}
for (int i=0;i<5;i++){
		A = (int*)calloc(lengths[i]*lengths[i],sizeof(int));
		disabledCols = (bool*)calloc(lengths[i],sizeof(bool));
		createMockArray(A,lengths[i]);
		printf("A(%dx%d)----------------------\n",lengths[i],lengths[i]);
		double start = omp_get_wtime();
		printf("Cilk det = %f\n",serialDetCilk(A,disabledCols,0,lengths[i],lengths[i]));
		double finish = omp_get_wtime();
		printf("time = %f\n",finish-start);
		free(A);
		free(disabledCols);	
	}
	
}
