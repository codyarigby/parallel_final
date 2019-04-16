/******************************************************************************
* Author : Cody Rigby
* Description  :
*   CMT-Bone Kernel CUDA
* Last Revised : 02/09/19
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_profiler_api.h>

/* -------------------------- Primary Parameters --------------------------- */
// Number of elements in each direction
// entire grid space is cubic
// of size (GRID_DIM , GRID_DIM , GRID_DIM)
// grid is always 3 dimensional

#ifndef GRID_DIM
#define GRID_DIM 4
#endif

// Size of each element (cubic), ternix dim 
// (ELEMENT_SIZE,ELEMENT_SIZE,ELEMENT_SIZE)

#ifndef ELEMENT_SIZE
#define ELEMENT_SIZE 16
#endif

// Number of physics parameters tracked 
// mass, energy and the three components of momentum
#define PHYSICAL_PARAMS 1

/* ------------------------- Secondary Parameters -------------------------- */
#define MPI_DTYPE MPI_DOUBLE
#define CARTESIAN_DIMENSIONS 3


/* ------------------------- Calculated Parameters ------------------------- */
// total number of elements in the grid
#define TOTAL_ELEMENTS  (GRID_DIM * GRID_DIM * GRID_DIM)
// size of the face of a single element parameter
#define FACE_SIZE       (ELEMENT_SIZE * ELEMENT_SIZE)

#define RK 2

/* ------------------------------------------------------------------------ */
/* -------------------------- Type Definitions ---------------------------- */
/* ------------------------------------------------------------------------ */

/* dtype: internal data storage type for calculations
   ttype: type to use for representing time */
typedef double dtype;
typedef double ttype;

typedef struct {
  int size;
  dtype * V;
} vectortype, *vector;

typedef struct {
  int rows;
  int cols;
  dtype ** M;
} matrixtype, *matrix;

typedef struct {
  int rows;
  int cols;
  int layers;
  dtype *** T;
} ternixtype, *ternix;

typedef struct {
  ternix *B;
} elementtype, *element;


/* ------------------------- Enum for Kernel Operation ------------------------- */
enum kernel_op{dr,ds,dt};


__global__ void gpu_operation(		double  [ELEMENT_SIZE][ELEMENT_SIZE],
									double[][ELEMENT_SIZE][ELEMENT_SIZE],
									double[][ELEMENT_SIZE][ELEMENT_SIZE],
									double[][ELEMENT_SIZE][ELEMENT_SIZE],
									double[][ELEMENT_SIZE][ELEMENT_SIZE],
									double[][ELEMENT_SIZE][ELEMENT_SIZE],
									double[][ELEMENT_SIZE][ELEMENT_SIZE]
									);


// insert these into the main routine
// const dim3 blockSize(ELEMENT_SIZE);
// const dim3 gridSize(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);

ttype tdiff(struct timespec a, struct timespec b)
/* Find the time difference. */
{
  ttype dt = (( b.tv_sec - a.tv_sec ) + ( b.tv_nsec - a.tv_nsec ) / 1E9);
  return dt;
}

struct timespec now(){
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t;
}


/* ------------------------------------------------------------------------- */
/* -------------------------- Vector Functions ----------------------------- */
/* ------------------------------------------------------------------------- */

vector new_vector(int size)
/* Make a new 'vector' type and allocate memory for it. */
{
  vector X = (vector)malloc(sizeof(vectortype));
  X->size = size;
  X->V = (dtype*)malloc(sizeof( dtype * ) * size);
  return X;
}

void delete_vector(vector X)
/* Free up the memory allocated for the vector X. */
{
  free(X->V);
  free(X);
}

void random_fill_vector(vector X, dtype lower, dtype upper)
/* Fill a vector with random numbers over [lower, upper) */
{
  int i;
  for (i = 0; i < (X->size); i++) {
    X->V[i] = ((dtype) rand() / (RAND_MAX)) * (upper - lower + 1) + lower;
  }
}

vector new_random_vector(int size, dtype lower, dtype upper)
/* Return a newly-allocated random vector */
{
  vector X = new_vector(size);
  random_fill_vector(X, lower, upper);
  return X;
}

/* ------------------------------------------------------------------------- */
/* -------------------------- Matrix Functions ----------------------------- */
/* ------------------------------------------------------------------------- */

matrix new_matrix(int rows, int cols)
/* Make a new 'matrix' type and allocate memory. Use: A->M[row][column]. */
{
  int i;
  matrix A  = (matrix)malloc(sizeof(matrixtype));
  A->rows   = rows;
  A->cols   = cols;
  A->M      = (dtype**)malloc(sizeof( dtype * ) * rows);

  for (i = 0; i < rows; i++) {
    A->M[i] = (dtype*)malloc(sizeof( dtype * ) * cols);
  }

  return A;
}

void delete_matrix(matrix A)
/* Free up the memory allocated for the matrix A. */
{
  int row;
  for (row = 0; row<(A->rows); row++) { free(A->M[row]); }
  free(A->M);
  free(A);
}

void zero_matrix(matrix A)
/* Zero out the matrix A. */
{
  int row, col;
  for(row = 0; row<(A->rows); row++) {
    for(col = 0; col<(A->cols); col++) {
      A->M[row][col] = (dtype) 0;
    }
  }
}

void random_fill_matrix(matrix A, dtype lower, dtype upper)
/* Fill a matrix with random numbers over [lower, upper). */
{
  int row, col;
  for (row = 0; row < (A->rows); row++) {
    for (col = 0; col < (A->cols); col++) {
      A->M[row][col] = (dtype) rand() / RAND_MAX * (upper - lower + 1) + lower;
    }
  }
}

matrix new_random_matrix(int rows, int cols, dtype lower, dtype upper)
/* Return a newly-allocated random matrix. */
{
  matrix A = new_matrix(rows, cols);
  random_fill_matrix(A, lower, upper);
  return A;
}

/* ------------------------------------------------------------------------- */
/* -------------------------- Ternix Functions ----------------------------- */
/* ------------------------------------------------------------------------- */

ternix new_ternix(int rows, int cols, int layers)
/*
  Make a new 'ternix' type and allocate memory for it.
  Access is done by: A->T[row][column][layer].
*/
{
  int i, j;
  ternix A  = (ternix)malloc(sizeof(ternixtype));
  A->rows   = rows;
  A->cols   = cols;
  A->layers = layers;
  A->T      = (dtype***)malloc( sizeof( dtype * ) * rows );

  for (i = 0; i<rows; i++) {
    A->T[i]       = (dtype**)malloc( sizeof( dtype * ) * cols);
    for (j = 0; j<cols; j++) {
      A->T[i][j]  = (dtype*)malloc( sizeof( dtype ) * layers);
    }
  }

  return A;
}

void delete_ternix(ternix A)
/* Free up the memory allocated for the ternix A. */
{
  int row, col;
  for (row = 0; row<(A->rows); row++) {
    for (col = 0; col<(A->cols); col++) {
      free(A->T[row][col]);
    }
    free(A->T[row]);
  }
  free(A->T);
  free(A);
}

void zero_ternix(ternix A)
/* Zero out the ternix A. */
{
  int row, col, layer;
  for(row = 0; row<(A->rows); row++) {
    for(col = 0; col<(A->cols); col++) {
      for(layer = 0; layer<(A->layers); layer++) {
        A->T[row][col][layer] = (dtype) 0;
      }
    }
  }
}

ternix new_zero_ternix(int rows, int cols, int layers)
/* Return a newly-allocated zeroed ternix. */
{
  ternix A = new_ternix(rows, cols, layers);
  zero_ternix(A);
  return A;
}

void random_fill_ternix(ternix A, dtype lower, dtype upper)
/* Fill a ternix with random numbers over [lower, upper). */
{
  int row, col, layer;
  for (row = 0; row<(A->rows); row++) {
    for (col = 0; col<(A->cols); col++) {
      for(layer = 0; layer<(A->layers); layer++) {
        A->T[row][col][layer] = ((dtype) rand() / (RAND_MAX)) *
                                 (upper - lower + 1) + lower;
      }
    }
  }
}

ternix new_random_ternix(int rows, int cols, int layers,
                         dtype lower, dtype upper)
/* Return a random newly-allocated ternix. */
{
  ternix A = new_ternix(rows, cols, layers);
  random_fill_ternix(A, lower, upper);
  return A;
}

/* ------------------------------------------------------------------------- */
/* -------------------------- Element Functions ---------------------------- */
/* ------------------------------------------------------------------------- */

element new_random_element(dtype lower, dtype upper)
/* Return an element with PHYSICAL_PARAMTERS blocks of ELEMENT_SIZE,
   randomly filled with ternices over [lower, upper). */
// elements have multiple ternices
{                         
  int i;
  element A = (element)malloc(sizeof(elementtype));

  A->B = (ternix *)malloc(sizeof( ternix * ) * PHYSICAL_PARAMS);

  for (i = 0; i < PHYSICAL_PARAMS; i++) {
    A->B[i] = new_random_ternix(  ELEMENT_SIZE, 
                                  ELEMENT_SIZE, 
                                  ELEMENT_SIZE,
                                  lower, upper );
  }

  return A;
}

element new_zero_element()
/* Return an element with PHYSICAL_PARAMTERS blocks of ELEMENT_SIZE. */
{
  int i;
  element A = (element)malloc( sizeof(elementtype) );

  A->B = (ternix *)malloc( sizeof( ternix * ) * PHYSICAL_PARAMS );

  for (i = 0; i < PHYSICAL_PARAMS; i++) {
    A->B[i] = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  }
  return A;
}

void delete_element(element A)
/* Frees up the memory allocated for the element A. */
{
  int i;

  for (i = 0; i < PHYSICAL_PARAMS; i++) { delete_ternix( A->B[i] ); }

  free(A->B);
  free(A);
}


/* ------------------------------------------------------------------------- */
/* ------------------------ Faked CMT-Nek Operations ----------------------- */
/* ------------------------------------------------------------------------- */

void operation_dr(matrix A, ternix B, ternix C)
/* Perform the R axis derivative operation, with kernel A and result C. */
{
  zero_ternix(C);

  int k, j, i, g;

  for (k = 0; k < ELEMENT_SIZE; k++) {
    for (j = 0; j < ELEMENT_SIZE; j++) {
      for (i = 0; i < ELEMENT_SIZE; i++) {
        for (g = 0; g < ELEMENT_SIZE; g++) {
          C->T[i][j][k] += A->M[i][g] * B->T[g][j][k]; } } } }
}

void operation_ds(matrix A, ternix B, ternix C)
/* Perform the S axis derivative operation, with kernel A and result C. */
{
  zero_ternix(C);

  int k, j, i, g;

  for (k = 0; k < ELEMENT_SIZE; k++) {
    for (j = 0; j < ELEMENT_SIZE; j++) {
      for (i = 0; i < ELEMENT_SIZE; i++) {
        for (g = 0; g < ELEMENT_SIZE; g++) {
          C->T[i][j][k] += A->M[j][g] * B->T[i][g][k]; } } } }
}

void operation_dt(matrix A, ternix B, ternix C)
/* Perform the T axis derivative operation, with kernel A and result C. */
{
  zero_ternix(C);

  int k, j, i, g;

  for (k = 0; k < ELEMENT_SIZE; k++) {
    for (j = 0; j < ELEMENT_SIZE; j++) {
      for (i = 0; i < ELEMENT_SIZE; i++) {
        for (g = 0; g < ELEMENT_SIZE; g++) {
          C->T[i][j][k] += A->M[k][g] * B->T[i][j][g]; } } } }
}

void operation_conv(ternix Q, ternix *RX, ternix Hx, ternix Hy, ternix Hz,
                    ternix Ur, ternix Us, ternix Ut)
/* Given Q, produce UR, US, and UT by faked transformation. HX, HY, and HZ
   are temporary space. RX is the list of transformation ternices. */
{

  /* Generate three random constants. */
  dtype a  = ((dtype) rand() / (RAND_MAX));
  dtype d  = ((dtype) rand() / (RAND_MAX));
  dtype c  = ((dtype) rand() / (RAND_MAX));

  /* indexing variables */
  // i,j,k are for indexing through the grid
  // r is for 'stages' (idk what this means but it makes the whole computation stage loop three times)
  // b is for indexing throught the physical paramaters of the individual elements
  int k, j, i;


  /* First, make HX, HY, and HZ, which are used in the next step. */
  for (k = 0; k < ELEMENT_SIZE; k++) {
    for (j = 0; j < ELEMENT_SIZE; j++) {
      for (i = 0; i < ELEMENT_SIZE; i++) {
        Hx->T[i][j][k] = a * Q->T[i][j][k];
        Hy->T[i][j][k] = d * Q->T[i][j][k];
        Hz->T[i][j][k] = c * Q->T[i][j][k];
      }
    }
  }

  /* Then, produce our outputs using HX, HY, HZ, and RX. */
  for (k = 0; k < ELEMENT_SIZE; k++) {
    for (j = 0; j < ELEMENT_SIZE; j++) {
      for (i = 0; i < ELEMENT_SIZE; i++) {

        /* Generate UR. */
        Ur->T[i][j][k] = ( RX[0]->T[i][j][k] * Hx->T[i][j][k] +
                           RX[1]->T[i][j][k] * Hy->T[i][j][k] +
                           RX[2]->T[i][j][k] * Hz->T[i][j][k] );

        /* Generate US. */
        Us->T[i][j][k] = ( RX[3]->T[i][j][k] * Hx->T[i][j][k] +
                           RX[4]->T[i][j][k] * Hy->T[i][j][k] +
                           RX[5]->T[i][j][k] * Hz->T[i][j][k] );

        /* Generate UT. */
        Ut->T[i][j][k] = ( RX[6]->T[i][j][k] * Hx->T[i][j][k] +
                           RX[7]->T[i][j][k] * Hy->T[i][j][k] +
                           RX[8]->T[i][j][k] * Hz->T[i][j][k] );

      }
    }
  }
}

void operation_sum(ternix X, ternix Y, ternix Z, ternix R)
/* Add three ternices together and put the result in R. */
{
  int k, j, i;

  for (k = 0; k < ELEMENT_SIZE; k++) {
    for (j = 0; j < ELEMENT_SIZE; j++) {
      for (i = 0; i < ELEMENT_SIZE; i++) {
        R->T[i][j][k] = X->T[i][j][k] + Y->T[i][j][k] + Z->T[i][j][k];
      }
    }
  }
}

void operation_rk(ternix Q, ternix R)
/* Perform a faked Runge Kutta stage (no previous stage information used). */
{
  int k, j, i;

  for (k = 0; k < ELEMENT_SIZE; k++) {
    for (j = 0; j < ELEMENT_SIZE; j++) {
      for (i = 0; i < ELEMENT_SIZE; i++) {
        Q->T[i][j][k] = ( R->T[i][j][k] * 0.5 +
                          R->T[i][j][k] * 0.25 +
                          Q->T[i][j][k] * 0.5 );
      }
    }
  }
}

/* ------------------------------------------------------------------------- */
/* ------------------------ Kernel GPU Operation --------------------------- */
/* ------------------------------------------------------------------------- */


void kernel_operation(			matrix kernel, double * kernel_flat, double * kernel_flat_dev,
								ternix Ur, ternix Us, ternix Ut, 
								ternix Vr, ternix Vs, ternix Vt,
								double * Ur_flat, 		double * Us_flat, 		double * Ut_flat,
								double * Vr_flat, 		double * Vs_flat, 		double * Vt_flat,
								double * Ur_flat_dev, 	double * Us_flat_dev, 	double * Ut_flat_dev,
								double * Vr_flat_dev, 	double * Vs_flat_dev, 	double * Vt_flat_dev)
{

	/*---------------------- time and other variables -------------------*/
	float 	setup_t,kernel_t,cleanup_t;	
	int 	n, nn;
	struct 	timespec tA, tB;

	size_t element_size = sizeof(double)*ELEMENT_SIZE*ELEMENT_SIZE*ELEMENT_SIZE;
	size_t kernel_size  = sizeof(double)*ELEMENT_SIZE*ELEMENT_SIZE;


	/*---------------------- copy over input ternices -------------------*/
	tA = now();
	n=0;
	nn=0;
	for(int i=0; i<ELEMENT_SIZE;i++){
		for(int j=0; j<ELEMENT_SIZE;j++){
		  	kernel_flat[nn++] = kernel->M[i][j];  
			for(int k=0; k<ELEMENT_SIZE;k++){
				Ur_flat[n] = Ur->T[i][j][k];
				Us_flat[n] = Us->T[i][j][k];
				Ut_flat[n] = Ut->T[i][j][k];
				n++;
			}
		}
	}

	/*---------------------- cuda copy the inputs ---------------------*/
	cudaError_t err = cudaMemcpy(kernel_flat_dev,	kernel_flat,	kernel_size,  cudaMemcpyHostToDevice);
	err 			= cudaMemcpy(Ur_flat_dev,		Ur_flat,		element_size, cudaMemcpyHostToDevice);
	err 			= cudaMemcpy(Us_flat_dev,		Us_flat,		element_size, cudaMemcpyHostToDevice);
	err 			= cudaMemcpy(Ut_flat_dev,		Ut_flat,		element_size, cudaMemcpyHostToDevice);


	/* ---------------------- set up cuda dimensions ------------------*/
	const dim3 blockSize(ELEMENT_SIZE);
    const dim3 gridSize(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);


	/* ---------------------- CUDA Function Time ------------------*/
	tA = now();

	gpu_operation<<<blockSize,gridSize>>>(			(double(*)[ELEMENT_SIZE])&kernel_flat_dev[0],
	                                                (double(*)[ELEMENT_SIZE][ELEMENT_SIZE])&Ur_flat_dev[0],
	                                                (double(*)[ELEMENT_SIZE][ELEMENT_SIZE])&Us_flat_dev[0],
	                                                (double(*)[ELEMENT_SIZE][ELEMENT_SIZE])&Ut_flat_dev[0],
	                                               	(double(*)[ELEMENT_SIZE][ELEMENT_SIZE])&Vr_flat_dev[0],
	                                                (double(*)[ELEMENT_SIZE][ELEMENT_SIZE])&Vs_flat_dev[0],
	                                                (double(*)[ELEMENT_SIZE][ELEMENT_SIZE])&Vt_flat_dev[0]);
	cudaError_t errk = cudaDeviceSynchronize();

	tB 			 = now();
	kernel_t = tdiff(tA, tB);
	// printf( "cuda function duration: %.8f seconds \n", tdiff(tA, tB));



    /* ---------------------- Copy over resulting matrices ------------------*/
    //err = cudaMemcpy(Vr_flat,		Vr_flat_dev,		element_size, cudaMemcpyDeviceToHost);
	//err = cudaMemcpy(Vs_flat,		Vs_flat_dev,		element_size, cudaMemcpyDeviceToHost);
	//err = cudaMemcpy(Vt_flat,		Vt_flat_dev,		element_size, cudaMemcpyDeviceToHost);

    // tA = now();
    nn=0;
	for(int i=0; i<ELEMENT_SIZE;i++){
		for(int j=0; j<ELEMENT_SIZE;j++){
			for(int k=0; k<ELEMENT_SIZE;k++){
				Vr->T[i][j][k]=Vr_flat[0];
				Vs->T[i][j][k]=Vs_flat[nn];
				Vt->T[i][j][k]=Vt_flat[nn];
				nn++;
			}
		}
	}

    /* ---------------------- don't free the memory yet ------------------*/	
	// cudaFree(kernel_flat_dev);
	// cudaFree(element_in_flat_dev);
	// cudaFree(element_out_flat_dev);

	/* ---------------------- display the time ------------------*/	
	//float total_t = (setup_t+cleanup_t+kernel_t);
	//printf( "total : %.8f, setup : %.8f%, kernel : %.8f%,  cleanup : %.8f% \n", total_t, setup_t/total_t, kernel_t/total_t, cleanup_t/total_t);

}

/* -------------------------------------------------------------------------  */
/* ---------------------------- CUDA Functions ------------------------------ */
/* -------------------------------------------------------------------------  */

__global__ void gpu_operation(	double k[ELEMENT_SIZE][ELEMENT_SIZE],
									double Ur[][ELEMENT_SIZE][ELEMENT_SIZE],
									double Us[][ELEMENT_SIZE][ELEMENT_SIZE],
									double Ut[][ELEMENT_SIZE][ELEMENT_SIZE],
									double Vr[][ELEMENT_SIZE][ELEMENT_SIZE],
									double Vs[][ELEMENT_SIZE][ELEMENT_SIZE],
									double Vt[][ELEMENT_SIZE][ELEMENT_SIZE])
{
	/*----------------------retreive a row from the input ternix ----------------------------*/
	int blockId = 	blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId =	blockId * blockDim.x + threadIdx.x;

	/*  ----------------------- shared memory setup -----------------------------*/
	__shared__ double Ur_partial_sums[ELEMENT_SIZE];
	__shared__ double Us_partial_sums[ELEMENT_SIZE];
	__shared__ double Ut_partial_sums[ELEMENT_SIZE];
	__shared__ double kernel[ELEMENT_SIZE][ELEMENT_SIZE];

	for(int g=0;g<ELEMENT_SIZE;g++)
		kernel[threadId][g] = k[threadId][g];
	__syncthreads();


	/*-------------------------------------- Calculate the partial sums ---------------------------------*/
	Ur_partial_sums[threadId] = 0.0;
	Us_partial_sums[threadId] = 0.0;
	Ut_partial_sums[threadId] = 0.0;

	for(int g=0;g<ELEMENT_SIZE;g++){
		Ur_partial_sums[threadId] += kernel[threadId][g]*Ur[g][blockIdx.y][blockIdx.z];
		Us_partial_sums[threadId] += kernel[threadId][g]*Us[blockIdx.x][g][blockIdx.z];
		Ut_partial_sums[threadId] += kernel[threadId][g]*Ut[blockIdx.x][blockIdx.y][g];
	}
	__syncthreads();

	/*-------------------------------------- Accumulate the partial sums ---------------------------------*/
  	Vr[blockIdx.x][blockIdx.y][blockIdx.z] = 0.0;
  	Vt[blockIdx.x][blockIdx.y][blockIdx.z] = 0.0;
  	Vs[blockIdx.x][blockIdx.y][blockIdx.z] = 0.0;
	if(threadId == 0){
		for(int g=0;g<ELEMENT_SIZE;g++){
			Vr[blockIdx.x][blockIdx.y][blockIdx.z]+=Ur_partial_sums[g];
			Vt[blockIdx.x][blockIdx.y][blockIdx.z]+=Ut_partial_sums[g];
			Vs[blockIdx.x][blockIdx.y][blockIdx.z]+=Us_partial_sums[g];
		}
	}
	// done
}




/* ------------------------------------------------------------------------- */
/* ---------------------------- Main Function ------------------------------ */
/* ------------------------------------------------------------------------- */
int main (int argc, char *argv[])
{

  // /* ------------------------------ Dimensions Setup --------------------------- */  
  // char *a = argv[1];
  // char *b = argv[2];
  // int GRID_DIM 	   = atoi(a);
  // int ELEMENT_SIZE = atoi(b);

  /* ------------------------------ Memory Setup --------------------------- */
  struct timespec tA, tB;

  // begin the time measurements
  tA = now();

  srand( 11 );

  // Index variables: i,j,k for grid indexing
  // r for stages (i dont know what that means)
  // b for the physical parameters of an element
  int i, j, k, r, b;

  // create the grid space as a ternix of elements
  element elements_Q[ GRID_DIM ][ GRID_DIM ][ GRID_DIM ];

  // create the grid space for the output 
  // elements_R and elements_Q are seperate for now
  // that could change in the future
  element elements_R[ GRID_DIM ][ GRID_DIM ][ GRID_DIM ];

  // generate elements in the grid
  for (i = 0; i < GRID_DIM; i++) {
    for (j = 0; j < GRID_DIM; j++){
      for (k = 0; k < GRID_DIM; k++){ 
        elements_Q[i][j][k] = new_random_element(0, 10);
        elements_R[i][j][k] = new_zero_element();
      }
    }
  }

  // The same kernel is used for everything 
  // kernel used on all elements and is the same size as a element ternix face
  // the ternix's in an element have faces the same size as this matrix
  matrix kernel = new_random_matrix(ELEMENT_SIZE, ELEMENT_SIZE, -10, 10);

  // The same transformation ternix (RX) is used for all elements.
  // This is an approximation, there should be one for each element. 
  ternix RX[9];

  // fill of the intermediate ternix
  for (i = 0; i < 9; i++) {
    RX[i] = new_random_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE, -1, 1);
  }

  // Intermediate 3D structures: used in conv operation 
  ternix Hx = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Hy = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Hz = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);

  // Intermediate 3D structures: outputs of conv operation 
  ternix Ur = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Us = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Ut = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);

  // Intermediate 3D structures: outputs of derivative operations 
  ternix Vr = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Vs = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Vt = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);

  // report the time it took to allocate the memory for the above objects
  tB = now();
  // printf( "HOST MESSAGE   : Memory Allocation took, %.8f seconds \n",tdiff(tA, tB));

  /* ------------------------------- GPU mem setup ----------------------------- */

  	double *kernel_flat, *kernel_flat_dev, *Ur_flat, *Us_flat, *Ut_flat, *Vr_flat, *Vs_flat, *Vt_flat, *Ur_flat_dev, *Us_flat_dev, *Ut_flat_dev, *Vr_flat_dev, *Vs_flat_dev, *Vt_flat_dev;

	size_t element_size = sizeof(double)*ELEMENT_SIZE*ELEMENT_SIZE*ELEMENT_SIZE;
	size_t kernel_size  = sizeof(double)*ELEMENT_SIZE*ELEMENT_SIZE;

	// allocate flattened host memory
	kernel_flat 	= (double*)malloc(kernel_size);
	Ur_flat 		= (double*)malloc(element_size);
	Us_flat 		= (double*)malloc(element_size);
	Ut_flat 		= (double*)malloc(element_size);
	Vr_flat 		= (double*)malloc(element_size);
	Vs_flat 		= (double*)malloc(element_size);
	Vt_flat 		= (double*)malloc(element_size);


	// allocate memory for 'flattened' kernel in cuda
	cudaError_t err = 	cudaMalloc((void**)&kernel_flat_dev,		 kernel_size);
	err = 				cudaMalloc((void**)&Ur_flat_dev,			element_size);
	err = 				cudaMalloc((void**)&Us_flat_dev,			element_size);
	err = 				cudaMalloc((void**)&Ut_flat_dev,			element_size);
	err = 				cudaMalloc((void**)&Vr_flat_dev,			element_size);
	err = 				cudaMalloc((void**)&Vs_flat_dev,			element_size);
	err = 				cudaMalloc((void**)&Vt_flat_dev,			element_size);
 

  /* ----------------------------------------------------------------------- */
  /* ------------------------------- Main Loop ----------------------------- */
  /* ----------------------------------------------------------------------- */

	// TIME MEASUREMENT VARIABLES
	int 	flag 		= 0;
	float 	cuda_t 		= 0.0;
	float 	serial_t 	= 0.0;

  // For each of the three 'stages': 
  for (r = 0; r < RK; r++) {

    /* --------------------------- Serial Compute (A) --------------------------- */

    for ( i = 0; i < GRID_DIM; i++ ) {
    for ( j = 0; j < GRID_DIM; j++ ) {
    for ( k = 0; k < GRID_DIM; k++ ) {

        // For each block in the element: 
        for ( b = 0; b < PHYSICAL_PARAMS; b++ ) {

			// Generate Ur, Us, and Ut. 
			// take the curently indexed ternix from the current element 
			// make the corresponding transformations to generate Ur, Us, and Ut
			// which are the input ternix data to the three kernels.
			operation_conv(elements_Q[i][j][k]->B[b], RX, Hx, Hy, Hz, Ur, Us, Ut);

			/* --------------------------- CUDA Kernels Begin --------------------------- */
	        if(r==0){
		          tA = now();
		          kernel_operation(	kernel, kernel_flat, kernel_flat_dev,
		          					Ur, Us, Ut, Vr, Vs, Vt,
		          			        Ur_flat, 		Us_flat, 		Ut_flat,
		          					Vr_flat, 		Vs_flat, 		Vt_flat,
		          					Ur_flat_dev, 	Us_flat_dev, 	Ut_flat_dev,
		          					Vr_flat_dev, 	Vs_flat_dev, 	Vt_flat_dev);
		          tB 		 	=  now();
		          cuda_t 	   +=  tdiff(tA,tB);  
		          printf( "CUDA Step : grid index [%d, %d, %d] duration: %.8f seconds \n", i,j,k, tdiff(tA, tB)); 
	      	}
	      	else{
	      		tA 			 = now();
	      		operation_dr(kernel, Ur, Vr);
	          	operation_ds(kernel, Us, Vs);
	          	operation_dt(kernel, Ut, Vt);
	      		tB 			 = now();
	      		serial_t 	+= tdiff(tA,tB); 
	      		printf( "SERIAL Step : grid index [%d, %d, %d] duration: %.8f seconds \n", i,j,k, tdiff(tA, tB)); 
	      	}

			/* --------------------------- CUDA Kernels End --------------------------- */
			/* Add Vr, Vs, and Vt to make R. */
			// elements_R is the output.
			operation_sum( Vr, Vs, Vt, elements_R[i][j][k]->B[b] );
        }
    }
    }
    } 
	cudaFree(kernel_flat_dev);
	cudaFree(Ur_flat_dev);
	cudaFree(Us_flat_dev);
	cudaFree(Ut_flat_dev);
	cudaFree(Vr_flat_dev);
	cudaFree(Vs_flat_dev);
	cudaFree(Vt_flat_dev);


    /* --------------------------- Location of Communication (C) ------------------------------- */
    //  This section might need to approximate the surface integral calculations
    /* --------------------------- Compute (B) --------------------------- */

    /* For each element owned by this rank: */
    for ( i = 0; i < GRID_DIM; i++ ) {
    for ( j = 0; j < GRID_DIM; j++ ) {
    for ( k = 0; k < GRID_DIM; k++ ) {

      /* For each block in the element: */
      for ( b = 0; b < PHYSICAL_PARAMS; b++ ) {
        // Perform a fake Runge Kutta stage (without R from the last stage)
        // to obtain a new value of Q. 
          operation_rk(elements_R[i][j][k]->B[b], elements_Q[i][j][k]->B[b]);
      }
    }
    }
    }

    // tB = now(); printf( "Step : %d   duration: %.8f seconds \n",r,tdiff(tA, tB)); 
  } // for each stage (RK) ...  

  // final report
  printf( "CUDA FINAL   avg duration: %.8f seconds \n", cuda_t/( GRID_DIM* GRID_DIM* GRID_DIM*PHYSICAL_PARAMS));
  printf( "SERIAL FINAL avg duration: %.8f seconds \n", serial_t/( GRID_DIM* GRID_DIM* GRID_DIM*PHYSICAL_PARAMS)); 
  printf("speedup for average processing of dim %d : %f \n",ELEMENT_SIZE,  
  	((serial_t/( GRID_DIM* GRID_DIM* GRID_DIM*PHYSICAL_PARAMS))/(cuda_t/( GRID_DIM* GRID_DIM* GRID_DIM*PHYSICAL_PARAMS))));

  printf("CUDA FINAL   total duration: %.8f seconds \n", cuda_t);
  printf("SERIAL FINAL total duration: %.8f seconds \n", serial_t); 

  printf("speedup for total time for grid dim %d and element dim %d : %f \n",GRID_DIM,ELEMENT_SIZE,serial_t/cuda_t);

  /* ----------------------------------------------------------------------- */
  /* -------------------------------- Cleanup ------------------------------ */
  /* ----------------------------------------------------------------------- */

  for ( i = 0; i < GRID_DIM; i++ ) {
  for ( j = 0; j < GRID_DIM; j++ ) {
  for ( k = 0; k < GRID_DIM; k++ ) {
    delete_element(elements_Q[i][j][k]);
    delete_element(elements_R[i][j][k]);
  }
  }
  }

  delete_matrix(kernel);

  for (i = 0; i < 9; i++) {
    delete_ternix(RX[i]);
  }

  delete_ternix(Hx);
  delete_ternix(Hy);
  delete_ternix(Hz);
  delete_ternix(Ur);
  delete_ternix(Us);
  delete_ternix(Ut);
  delete_ternix(Vr);
  delete_ternix(Vs);
  delete_ternix(Vt);

  //printf("hello world");

  return 0;
}
