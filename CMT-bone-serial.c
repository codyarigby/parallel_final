/*
  A pseudo-representative application to model a serial version of NEK.
    Authors : Cody Rigby
    Class   : EEL6763

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#include <time.h>

/* -------------------------- Primary Parameters --------------------------- */

// Number of elements in each direction
// entire grid space is cubic
// of size (GRID_DIM , GRID_DIM , GRID_DIM)
// grid is always 3 dimensional
#ifndef GRID_DIM
#define GRID_DIM 8
#endif


// Size of each element (cubic), ternix dim 
// (ELEMENT_SIZE,ELEMENT_SIZE,ELEMENT_SIZE)
#ifndef ELEMENT_SIZE
#define ELEMENT_SIZE 5
#endif

// Number of physics parameters tracked 
// mass, energy and the three components of momentum
#define PHYSICAL_PARAMS 5

/* ------------------------- Secondary Parameters -------------------------- */

#define MPI_DTYPE MPI_DOUBLE
#define CARTESIAN_DIMENSIONS 3


/* ------------------------- Calculated Parameters ------------------------- */

// total number of elements in the grid
#define TOTAL_ELEMENTS  (GRID_DIM* GRID_DIM * GRID_DIM)
// size of the face of a single element parameter
#define FACE_SIZE       (ELEMENT_SIZE * ELEMENT_SIZE)

#define RK 3


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

/* NOTE: the above definition does not know the number of physical parameters,
   so it is dependent on the macro PHYSICAL_PARAMS for the size of B. */

/* ------------------------------------------------------------------------ */
/* ------------------------- Utility Functions ---------------------------- */
/* ------------------------------------------------------------------------ */

ttype tdiff(struct timespec a, struct timespec b)
/* Find the time difference. */
{
  ttype dt = (( b.tv_sec - a.tv_sec ) + ( b.tv_nsec - a.tv_nsec ) / 1E9);
  return dt;
}

struct timespec now()
/* Return the current time. */
{
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
  vector X = malloc(sizeof(vectortype));
  X->size = size;
  X->V = malloc(sizeof( dtype * ) * size);
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
  matrix A = malloc(sizeof(matrixtype));
  A->rows = rows;
  A->cols = cols;
  A->M = malloc(sizeof( dtype * ) * rows);

  for (i = 0; i < rows; i++) {
    A->M[i] = malloc(sizeof( dtype * ) * cols);
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
  ternix A = malloc(sizeof(ternixtype));
  A->rows = rows;
  A->cols = cols;
  A->layers = layers;
  A->T = malloc( sizeof( dtype * ) * rows );

  for (i = 0; i<rows; i++) {
    A->T[i] = malloc( sizeof( dtype * ) * cols);
    for (j = 0; j<cols; j++) {
      A->T[i][j] = malloc( sizeof( dtype ) * layers);
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
  element A = malloc(sizeof(elementtype));

  A->B = malloc(sizeof( ternix * ) * PHYSICAL_PARAMS);

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
  element A = malloc( sizeof(elementtype) );

  A->B = malloc( sizeof( ternix * ) * PHYSICAL_PARAMS );

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
  dtype bb = ((dtype) rand() / (RAND_MAX));
  dtype c  = ((dtype) rand() / (RAND_MAX));

  int k, j, i, r, b;

  /* First, make HX, HY, and HZ, which are used in the next step. */

  for (k = 0; k < ELEMENT_SIZE; k++) {
    for (j = 0; j < ELEMENT_SIZE; j++) {
      for (i = 0; i < ELEMENT_SIZE; i++) {
        Hx->T[i][j][k] = a * Q->T[i][j][k];
        Hy->T[i][j][k] = bb * Q->T[i][j][k];
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
/* ---------------------------- Main Function ------------------------------ */
/* ------------------------------------------------------------------------- */

int main (int argc, char *argv[])
{


  /* ------------------------------ Memory Setup --------------------------- */

  struct timespec tA, tB;

  // begin the time measurements
  tA = now();

  srand( 11 );

  /* Index variables: {generic, timestep, rk-index, element, block, axis} */
  int i, j, k, r, b;

  // create the grid space as a ternix of elements
  element elements_Q[ GRID_DIM ][  GRID_DIM ][ GRID_DIM ];

  //
  element elements_R[ GRID_DIM ][  GRID_DIM ][ GRID_DIM ];

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
  printf( "Memory Allocation took, %.8f seconds \n",tdiff(tA, tB));

  /* ----------------------------------------------------------------------- */
  /* ------------------------------- Main Loop ----------------------------- */
  /* ----------------------------------------------------------------------- */


  tA = now();

  // For each of the three 'stages': 
  for (r = 0; r < RK; r++) {

    /* --------------------------- Serial Compute (A) --------------------------- */


    // For each element in the entire grid
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

          /* Perform the three derivative computations (R, S, T). */
          // the results are in the ternixes Vr,Vs,Vt
          operation_dr(kernel, Ur, Vr);
          operation_ds(kernel, Us, Vs);
          operation_dt(kernel, Ut, Vt);

          /* Add Vr, Vs, and Vt to make R. */
          // elements_R is the output.
          operation_sum( Vr, Vs, Vt, elements_R[i][j][k]->B[b] );

        }
    }
    }
    } 

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

    
    tB = now(); printf( "Step : %d   duration: %.8f seconds \n",r,tdiff(tA, tB)); 

  } // for each stage (RK) ...  

  /* ----------------------------------------------------------------------- */
  /* -------------------------------- Cleanup ------------------------------ */
  /* ----------------------------------------------------------------------- */

  // start time for cleanup
  tA = now();

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

  printf( "Cleanup took %.8f seconds \n", tdiff(tA, tB)); 

  return 0;
}