#include <petsc.h>
#include <petscksp.h>  //KSP solver library
#include <math.h>
#include <mpi.h>
#include "petscvec.h"  

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **args)
{
	Vec				 u, b ;			       // Numerical solution, RHS
	Mat				 A;		               // Coefficient Matrix
	KSP				 ksp;                          // Linear solver context 
	PC				 pc;                           // Preconditioner context 
	PetscReal		 norm, normb;                          // To get relative residue
	PetscErrorCode	 ierr;
	PetscInt		 i, j, nloc;  
	PetscInt		 N = 64, its, n;		     // Default size of A (aka the Dummy data-set)
	PetscScalar      pi = 4 * atan(1.0);
	//PetscViewer	 viewer, lab;		                     // To print soln vector to text file
	DMBoundaryType   bx = DM_BOUNDARY_NONE, by = DM_BOUNDARY_NONE;
	DMDAStencilType  stype = DMDA_STENCIL_STAR;                  // five-pt stencil for five-pt laplacian

	PetscInitialize(&argc, &args, (char*)0, 0);
	ierr = PetscOptionsGetInt(NULL, "-n", &n, NULL); CHKERRQ(ierr); 

	// Initialize MPI and get number of processes and my number or rank
	int numprocs, myid;
	double startTime, endTime;
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	// Compute the matrix and right-hand-side vector that define the linear system, Au = b.
	DM	   da;              // DM object
	nloc = pow(N, 0.5);     // size of interior mesh

	// Create 2D grid
	ierr = DMDACreate2d(PETSC_COMM_WORLD, bx, by, stype, -nloc, -nloc, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
	// Or input global grid size at command line using for example: -da_grid_x 64 -da_grid_y 64

	ierr = DMSetUp(da); CHKERRQ(ierr);
	ierr = DMSetFromOptions(da); CHKERRQ(ierr);

	// Define the unit square over which we are solving (note: 2D problem ignores z values)
	ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0); CHKERRQ(ierr);  
	DMDALocalInfo     info;					
	ierr = DMDAGetLocalInfo(da, &info);

	// Display information about the 2D grid
	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nNumber of local grid points per individual process\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "local grid points xm = %d\n", info.xm); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "local grid points ym = %d\n", info.ym); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nNumber of global grid points\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Global grid points mx = %d\n", info.mx); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Global grid points my = %d\n", info.my); CHKERRQ(ierr);

	//mx is global number of grid points in x direction
	PetscScalar hx = 1.0 / (info.mx+1);		   // x grid spacing 
	PetscScalar hy = 1.0 / (info.my+1);                // y grid spacing 

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nThe grid spacing is: \n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "hx is = %.5f\n", hx); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "hy is = %.5f\n", hy); CHKERRQ(ierr);

	// Processor zero starts its clock
	if (myid == 0)
	{
		startTime = MPI_Wtime();
	}

	// Create DM Matrix of type mpiaij (default)
	// Begin creating a dummy data-set
	ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);

	// Set matrix values
	PetscInt		ncols;
	PetscScalar		v[5];
	MatStencil		col[5], row;  // stores grid coordinates over the 2D grid 

	// Set matrix elements for the 2-D, five-point stencil in parallel.
	// Taking advantage of the stencil structure of the DMDA to write into my matrix
	for (j = info.ys; j < info.ys + info.ym; j++){   	// xs and ys are starting points of this processor
		for (i = info.xs; i < info.xs + info.xm; i++) {
			ncols = 0;
			row.i = i;					// indexing is global (across the entire 2D grid)
			row.j = j;    
			if (i > 0) {
				col[ncols].i = i - 1;  
				col[ncols].j = j;
				v[ncols++] = 1.0;		// sub diagonal
			}
			if (i < info.mx-1) {
				col[ncols].i = i + 1;
				col[ncols].j = j;
				v[ncols++] = 1.0;		// super diagonal
			}
			if (j > 0) {
				col[ncols].i = i;
				col[ncols].j = j - 1;
				v[ncols++] = 1.0;		// other lower diagonal
			}
			if (j < info.my - 1) {
				col[ncols].i = i;
				col[ncols].j = j+1;
				v[ncols++] = 1.0;		// other upper diagonal
			}
			col[ncols].i = i;
			col[ncols].j = j;
			v[ncols++] = -4.0;           // main diagonal entries
			ierr = MatSetValuesStencil(A, 1, &row, ncols, col, v, INSERT_VALUES);  CHKERRQ(ierr);
		}
	}
	// Assemble the matrix 
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	// Create vectors.  
	ierr = DMCreateGlobalVector(da, &u);  CHKERRQ(ierr);    // Solution vector
	ierr = DMCreateGlobalVector(da, &b);  CHKERRQ(ierr);   
	PetscScalar   **array;

	ierr = DMDAVecGetArray(da, b, &array); CHKERRQ(ierr);
	for (j = info.ys; j < info.ys + info.ym; j++) {   	
		for (i = info.xs; i < info.xs + info.xm; i++) {
			array[j][i] = -20*pi*pi*hx*hx*sin(2 * pi*(j+1)*hx)*sin(4 * pi*(i+1)*hx);
		}
	}
	ierr = DMDAVecRestoreArray(da, b, &array); CHKERRQ(ierr);
	ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

	//printf("\n rhs vector is: \n");
	//ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	//printf("\n Matrix A IS: \n");
	//ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	// Creates linear solver context
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);  // Preconditioning matrix is A
	ierr = KSPSetType(ksp, KSPGMRES);                  // Preconditioned GMRES Method
	ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);  
	ierr = PCSetType(pc, PCGAMG); CHKERRQ(ierr);       // GAMG Preconditioner
	ierr = KSPSetTolerances(ksp, 1e-07, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
	ierr = PCSetFromOptions(pc); CHKERRQ(ierr);

	//Solve linear system
	ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);  // u is solution vector : n x 1

	// Stop Timer
	if (myid == 0)
	{
		endTime = MPI_Wtime();
		ierr = PetscPrintf(PETSC_COMM_WORLD, "\nRuntime is = %.16f\n", endTime-startTime); CHKERRQ(ierr);
	}

	// Print solution vector of 64-by-64 matrix to text file
	char	name[] = "soln_64by64";
	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &lab); CHKERRQ(ierr);
	ierr = PetscViewerSetType(lab, PETSCVIEWERASCII);  CHKERRQ(ierr);
	ierr = PetscViewerFileSetMode(lab, FILE_MODE_WRITE);  CHKERRQ(ierr);
	ierr = PetscViewerFileSetName(lab, name);  CHKERRQ(ierr);
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "soln_64by64.out", &viewer);  CHKERRQ(ierr);
	ierr = VecView(u, viewer); CHKERRQ(ierr);
	
	ierr = KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	// Show the residue and total no. of iterations
	ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
	ierr = VecNorm(b, NORM_2, &normb); CHKERRQ(ierr);
	ierr = KSPGetResidualNorm(ksp, &norm); CHKERRQ(ierr);

	// print out norm of residual
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual = %e, Iterations = %D\n\n", norm, its); CHKERRQ(ierr);

	// Destroy PETSc objects
	ierr = VecDestroy(&u); CHKERRQ(ierr);
	ierr = VecDestroy(&b); CHKERRQ(ierr); 
	ierr = MatDestroy(&A); CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	ierr = PetscFinalize();						// Finalizes Petsc Libary as well as MPI
	return 0;
}
