/*
  A pseudo-representative application to model NEK.
    Copyright (C) 2016  { Dylan Rudolph, NSF CHREC, UF CCMT }
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
#include <mpi.h>

#include "time.h"

/* -------------------------- Primary Parameters --------------------------- */

/* The rank which shows its timing output */
#ifndef PROBED_RANK
#define PROBED_RANK 0
#endif

/* Number of simulation timesteps */
#ifndef TIMESTEPS
#define TIMESTEPS 3
#endif

/* Number of processes in each dimension (must all be even numbers)
   MPI must be run with (product of these numbers) processes. */
#ifndef CARTESIAN_X 
#define CARTESIAN_X 2 
#endif

#ifndef CARTESIAN_Y
#define CARTESIAN_Y 2
#endif

#ifndef CARTESIAN_Z
#define CARTESIAN_Z 2
#endif

/* Number of elements per process in each dimension */
#ifndef ELEMENTS_X
#define ELEMENTS_X 2
#endif

#ifndef ELEMENTS_Y
#define ELEMENTS_Y 2
#endif

#ifndef ELEMENTS_Z
#define ELEMENTS_Z 2
#endif

/* Size of each element (cubic) */
#ifndef ELEMENT_SIZE
#define ELEMENT_SIZE 5
#endif

/* Number of physics parameters tracked */
#define PHYSICAL_PARAMS 5

/* ------------------------- Secondary Parameters -------------------------- */

#define MPI_DTYPE MPI_DOUBLE
#define CARTESIAN_DIMENSIONS 3
#define CARTESIAN_REORDER 0
#define CARTESIAN_WRAP {0, 0, 0}
#define RK 3

/* ------------------------- Calculated Parameters ------------------------- */

#define ELEMENTS_PER_PROCESS (ELEMENTS_X * ELEMENTS_Y * ELEMENTS_Z)
#define ELEMENTS_ON_X_FACE (ELEMENTS_Y * ELEMENTS_Z)
#define ELEMENTS_ON_Y_FACE (ELEMENTS_X * ELEMENTS_Z)
#define ELEMENTS_ON_Z_FACE (ELEMENTS_X * ELEMENTS_Y)
#define FACE_SIZE (ELEMENT_SIZE * ELEMENT_SIZE)

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
{
  int i;
  element A = malloc(sizeof(elementtype));

  A->B = malloc(sizeof( ternix * ) * PHYSICAL_PARAMS);

  for (i = 0; i < PHYSICAL_PARAMS; i++) {
    A->B[i] = new_random_ternix( ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE,
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
/* ---------------------------- Face Functions ----------------------------- */
/* ------------------------------------------------------------------------- */

vector new_extracted_faces(element *elements, int axis, int sign)
/* Return a collection of faces from a set of elements, where the faces
   for each physical parameter have been clumped together in anticipation
   of a transfer operation. Possible values for arguments:
     axis:  {0, 1, 2}  |  X, Y, or Z
     sign:  {-1, 1}    |  Minus or Plus Face
   The resulting vector output is of size: PHYSICAL_PARAMS * FACE_SIZE
   multiplied by the number of elements on the face of interest. */
{
  int i, b, e, row, col, layer, plane, EoF;

  switch (axis) { /* EoF: elements on face */
  case 0: EoF = ELEMENTS_ON_X_FACE; break;
  case 1: EoF = ELEMENTS_ON_Y_FACE; break;
  case 2: EoF = ELEMENTS_ON_Z_FACE; break; }

  vector faces = new_vector(EoF * PHYSICAL_PARAMS * FACE_SIZE);

  /* ---------------------------- Extraction ------------------------------- */

  /* The plane is the index of the lower or upper face. */
  plane = (sign > 0) ? ELEMENT_SIZE - 1 : 0;

  /* Index in the output vector */
  i = 0;

  /* For each element owned by this rank: */
  for (e = 0; e < EoF; e++) {

    /* For each block in the element: */
    for (b = 0; b < PHYSICAL_PARAMS; b++) {

      if ( axis == 0 ) {
        /* If this is the X axis, the plane is on the row dimension. */

        for (col = 0; col < ELEMENT_SIZE; col++) {
          for (layer = 0; layer < ELEMENT_SIZE; layer++) {
            faces->V[i] = elements[e]->B[b]->T[plane][col][layer]; i++; } }

      } else if ( axis == 1 ) {
        /* If this is the Y axis, the plane is on the column dimension. */

        for (row = 0; row < ELEMENT_SIZE; row++) {
          for (layer = 0; layer < ELEMENT_SIZE; layer++) {
            faces->V[i] = elements[e]->B[b]->T[row][plane][layer]; i++; } }

      } else if ( axis == 2 ) {
        /* If this is the Z axis, the plane is on the layer dimension. */

        for (row = 0; row < ELEMENT_SIZE; row++) {
          for (col = 0; col < ELEMENT_SIZE; col++) {
            faces->V[i] = elements[e]->B[b]->T[row][col][plane]; i++; } }
      }
    }
  }

  return faces;
}

vector new_empty_faces(int axis)
/* Same as above, but intended for the recv side, so not initialized. */
{
  int EoF;

  switch (axis) { /* EoF: elements on face */
  case 0: EoF = ELEMENTS_ON_X_FACE; break;
  case 1: EoF = ELEMENTS_ON_Y_FACE; break;
  case 2: EoF = ELEMENTS_ON_Z_FACE; break; }

  return new_vector(EoF * PHYSICAL_PARAMS * FACE_SIZE);
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

  dtype a = ((dtype) rand() / (RAND_MAX));
  dtype b = ((dtype) rand() / (RAND_MAX));
  dtype c = ((dtype) rand() / (RAND_MAX));

  int k, j, i;

  /* First, make HX, HY, and HZ, which are used in the next step. */

  for (k = 0; k < ELEMENT_SIZE; k++) {
    for (j = 0; j < ELEMENT_SIZE; j++) {
      for (i = 0; i < ELEMENT_SIZE; i++) {
        Hx->T[i][j][k] = a * Q->T[i][j][k];
        Hy->T[i][j][k] = b * Q->T[i][j][k];
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

        /* Generate UR. */
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
  /* ------------------------------- MPI Setup------------------------------ */

  MPI_Init(&argc, &argv);

  int rank, comrades;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comrades);

  int cart_sizes[CARTESIAN_DIMENSIONS] = {CARTESIAN_X,CARTESIAN_Y,CARTESIAN_Z};
  int cart_wrap[CARTESIAN_DIMENSIONS] = CARTESIAN_WRAP;

  MPI_Comm cart_comm;
  MPI_Cart_create( MPI_COMM_WORLD, CARTESIAN_DIMENSIONS, cart_sizes, cart_wrap,
                   CARTESIAN_REORDER, &cart_comm );

  /* ------------------------------ Memory Setup --------------------------- */

  struct timespec tA, tB;

  if (rank == PROBED_RANK) tA = now();

  srand( 11 );

  /* Index variables: {generic, timestep, rk-index, element, block, axis} */
  int i, t, r, e, b, axis;

  element elements_Q[ ELEMENTS_PER_PROCESS ];
  element elements_R[ ELEMENTS_PER_PROCESS ];

  for (e = 0; e < ELEMENTS_PER_PROCESS; e++) {
    elements_Q[e] = new_random_element(0, 10);
    elements_R[e] = new_zero_element();
  }

  /* The same kernel is used for everything */
  matrix kernel = new_random_matrix(ELEMENT_SIZE, ELEMENT_SIZE, -10, 10);

  /* The same transformation ternix (RX) is used for all elements.
     This is an approximation, there should be one for each element. */
  ternix RX[9];

  for (i = 0; i < 9; i++) {
    RX[i] = new_random_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE, -1, 1);
  }

  /* Intermediate 3D structures: used in conv operation */
  ternix Hx = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Hy = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Hz = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);

  /* Intermediate 3D structures: outputs of conv operation */
  ternix Ur = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Us = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Ut = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);

  /* Intermediate 3D structures: outputs of derivative operations */
  ternix Vr = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Vs = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);
  ternix Vt = new_zero_ternix(ELEMENT_SIZE, ELEMENT_SIZE, ELEMENT_SIZE);

  if (rank == PROBED_RANK) { tB = now(); printf( "Allocation,%.8f\n",
                                                 tdiff(tA, tB)); }

  /* ----------------------------------------------------------------------- */
  /* ------------------------------- Main Loop ----------------------------- */
  /* ----------------------------------------------------------------------- */

  /* For each timestep: */
  for ( t = 0; t < TIMESTEPS; t++ ) {

    if (rank == PROBED_RANK) tA = now();

    /* For each of the three 'stages': */
    for (r = 0; r < RK; r++) {

      /* --------------------------- Compute (A) --------------------------- */

      /* For each element owned by this rank: */
      for ( e = 0; e < ELEMENTS_PER_PROCESS; e++ ) {

        /* For each block in the element: */
        for ( b = 0; b < PHYSICAL_PARAMS; b++ ) {

          /* Generate Ur, Us, and Ut. */
          operation_conv(elements_Q[e]->B[b], RX, Hx, Hy, Hz, Ur, Us, Ut);

          /* Perform the three derivative computations (R, S, T). */
          operation_dr(kernel, Ur, Vr);
          operation_ds(kernel, Us, Vs);
          operation_dt(kernel, Ut, Vt);

          /* Add Vr, Vs, and Vt to make R. */
          operation_sum( Vr, Vs, Vt, elements_R[e]->B[b] );

        }
      }

      /* --------------------------- Communicate --------------------------- */

      /* above: plus neighbor, below: minus neighbor, index along this axis */
      int above, below, index;

      /* Unused status flag */
      MPI_Status status;

      /* Cartesian coordinates */
      int coords[CARTESIAN_DIMENSIONS];

      /* Determine our location in the cartesian grid. */
      MPI_Cart_coords(cart_comm, rank, CARTESIAN_DIMENSIONS, coords);

      vector above_faces_to_send, above_faces_to_recv;
      vector below_faces_to_send, below_faces_to_recv;

      for ( axis = 0; axis < CARTESIAN_DIMENSIONS; axis++ ) {

        /* Find our index along this axis. */
        index = coords[axis];

        /* Determine our neighbors. */
        MPI_Cart_shift(cart_comm, axis, 1, &below, &above);

        /* --------------------------- Transfers --------------------------- */

        /* Significant operations are given a heading, everything else is just
           instrumentation and logging. */

        /* ------------------------ Even Axis Index ------------------------ */

        if ( (index % 2) == 0 ) {

          /* If my index on this axis is even:
             - SEND  faces to    ABOVE  neighbor  (23)
             - RECV  faces from  ABOVE  neighbor  (47)
             - SEND  faces to    BELOW  neighbor  (61)
             - RECV  faces from  BELOW  neighbor  (73) */

          if ( above != MPI_PROC_NULL ) {

            /* - - - - - - - - - - - - Prepare Faces - - - - - - - - - - - - */
            above_faces_to_send = new_extracted_faces(elements_R, axis, 1);
            above_faces_to_recv = new_empty_faces(axis);
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - - Send Above  - - - - - - - - - - - - */
            MPI_Send( above_faces_to_send->V, above_faces_to_send->size,
                      MPI_DTYPE, above, 23, cart_comm );
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - - Recv Above  - - - - - - - - - - - - */
            MPI_Recv( above_faces_to_recv->V, above_faces_to_recv->size,
                      MPI_DTYPE, above, 47, cart_comm, &status );
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - Cleanup Faces - - - - - - - - - - - - */
            delete_vector(above_faces_to_send);
            delete_vector(above_faces_to_recv);
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

          }

          if ( below != MPI_PROC_NULL ) {

            /* - - - - - - - - - - - - Prepare Faces - - - - - - - - - - - - */
            below_faces_to_send = new_extracted_faces(elements_R, axis, -1);
            below_faces_to_recv = new_empty_faces(axis);
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - - Send Below  - - - - - - - - - - - - */
            MPI_Send( below_faces_to_send->V, below_faces_to_send->size,
                      MPI_DTYPE, below, 61, cart_comm );
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - - Recv Below  - - - - - - - - - - - - */
            MPI_Recv( below_faces_to_recv->V, below_faces_to_recv->size,
                      MPI_DTYPE, below, 73, cart_comm, &status );
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - Cleanup Faces - - - - - - - - - - - - */
            delete_vector(below_faces_to_send);
            delete_vector(below_faces_to_recv);
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

          }
        }

        /* ------------------------- Odd Axis Index ------------------------ */

        else {

          /* If my index on this axis is odd:
             - RECV  faces from  BELOW  neighbor  (23)
             - SEND  faces from  BELOW  neighbor  (47)
             - RECV  faces to    ABOVE  neighbor  (61)
             - SEND  faces from  ABOVE  neighbor  (73) */

          if ( below != MPI_PROC_NULL ) {

            /* - - - - - - - - - - - - Prepare Faces - - - - - - - - - - - - */
            below_faces_to_send = new_extracted_faces(elements_R, axis, -1);
            below_faces_to_recv = new_empty_faces(axis);
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - - Recv Below  - - - - - - - - - - - - */
            MPI_Recv( below_faces_to_recv->V, below_faces_to_recv->size,
                      MPI_DTYPE, below, 23, cart_comm, &status );
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - - Send Below  - - - - - - - - - - - - */
            MPI_Send( below_faces_to_send->V, below_faces_to_send->size,
                      MPI_DTYPE, below, 47, cart_comm );
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - Cleanup Faces - - - - - - - - - - - - */
            delete_vector(below_faces_to_send);
            delete_vector(below_faces_to_recv);
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

          }

          if ( above != MPI_PROC_NULL ) {

            /* - - - - - - - - - - - - Prepare Faces - - - - - - - - - - - - */
            above_faces_to_send = new_extracted_faces(elements_R, axis, 1);
            above_faces_to_recv = new_empty_faces(axis);
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - - Recv Above  - - - - - - - - - - - - */
            MPI_Recv( above_faces_to_recv->V, above_faces_to_recv->size,
                      MPI_DTYPE, above, 61, cart_comm, &status );
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - - Send Above  - - - - - - - - - - - - */
            MPI_Send( above_faces_to_send->V, above_faces_to_send->size,
                      MPI_DTYPE, above, 73, cart_comm );
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

            /* - - - - - - - - - - - - Cleanup Faces - - - - - - - - - - - - */
            delete_vector(above_faces_to_send);
            delete_vector(above_faces_to_recv);
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

          }

        }

      } /* for each axis ... */

      /* --------------------------- Compute (B) --------------------------- */

      /* For each element owned by this rank: */
      for ( e = 0; e < ELEMENTS_PER_PROCESS; e++ ) {

        /* For each block in the element: */
        for ( b = 0; b < PHYSICAL_PARAMS; b++ ) {

          /* Perform a fake Runge Kutta stage (without R from the last stage)
             to obtain a new value of Q. */
          operation_rk(elements_R[e]->B[b], elements_Q[e]->B[b]);

        }
      }

    } /* For each stage ... */

    if (rank == PROBED_RANK) { tB = now(); printf( "Step,%.8f\n",
                                                   tdiff(tA, tB) ); }

  } /* for each timestep ... */

  /* ----------------------------------------------------------------------- */
  /* -------------------------------- Cleanup ------------------------------ */
  /* ----------------------------------------------------------------------- */

  if (rank == PROBED_RANK) tA = now();

  for (e = 0; e < ELEMENTS_PER_PROCESS; e++) {
    delete_element(elements_Q[e]);
    delete_element(elements_R[e]);
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

  if (rank == PROBED_RANK) { tB = now(); printf( "Cleanup,%.8f\n",
                                                 tdiff(tA, tB)); }

  MPI_Finalize();

  return 0;
}