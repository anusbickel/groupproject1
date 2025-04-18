#include "domain.h"
#include "gf.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

static void fill_guess_values(struct ngfs_2d *gfs);
static void exchange_ghost_cells(struct ngfs_2d *gfs);
static void print_gf_function(struct ngfs_2d *gfs);
void gauss_seidel_solver_2d(struct ngfs_2d *gfs, int max_iters);
int MPI_Isendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                  MPI_Request *request);
double check_error(struct ngfs_2d *gfs);


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size = -1; 
    int mpi_rank = -1; 
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc != 5)
    {   
        fprintf(stderr, "Usage: %s NX NY PX PY\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }   

    const int global_nx = atoi(argv[1]);
    const int global_ny = atoi(argv[2]);
    const int px = atoi(argv[3]);
    const int py = atoi(argv[4]);

    if (global_nx <= 0 || global_ny <= 0 || px <= 0 || py <= 0)
    {
        fprintf(stderr, "NX, NY, PX, PY all > 0 required (%d, %d, %d, %d)\n",
                global_nx, global_ny, px, py);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (px * py != mpi_size)
    {
        fprintf(stderr, "PX * PY != MPI_SIZE (%d, %d, %d)\n", px, py, mpi_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const double dx = 1.0 / (global_nx - 1);
    const double dy = 1.0 / (global_ny - 1);
    const double global_x0 = 0.0;
    const double global_y0 = 0.0;

    const int gs = 1;

    struct ngfs_2d gfs;
    gfs.vars = NULL;

    const int nvars = 3; // uval 0, sval 1, eval 2

    printf("SETTING UP DOMAIN\n");
    setup_2d_domain(px, py, mpi_rank, global_nx, global_ny, gs, global_x0,
                    global_y0, dx, dy, &gfs.domain);

    printf("ALLOCATING\n");
    ngfs_2d_allocate(nvars, &gfs);

    printf("FILLING VALUES\n");
    fill_guess_values(&gfs);
    MPI_Barrier(MPI_COMM_WORLD);

    //exchange_ghost_cells(&gfs);
    //MPI_Barrier(MPI_COMM_WORLD);

    gauss_seidel_solver_2d(&gfs, 10240);

    print_gf_function(&gfs);

    ngfs_deallocate_2d(&gfs);

    MPI_Finalize();
    return 0;
}


void gauss_seidel_solver_2d(struct ngfs_2d *gfs, int check_every)
{
    int left_rank = gfs->domain.lower_x_rank;
    int right_rank = gfs->domain.upper_x_rank;
    int bottom_rank = gfs->domain.lower_y_rank;
    int top_rank = gfs->domain.upper_y_rank;

    double *uval = gfs->vars[0]->val;
    double *sval = gfs->vars[1]->val;

    int ystart; int yend; int xstart; int xend;

    if (top_rank == INVALID_RANK)    {yend   = gfs->ny;} else {yend   = gfs->ny-1;}
    if (bottom_rank == INVALID_RANK) {ystart = 0;}       else {ystart = 1;}
    if (left_rank == INVALID_RANK)   {xstart = 0;}       else {xstart = 1;}
    if (right_rank == INVALID_RANK)  {xend   = gfs->nx;} else {xend   = gfs->nx-1;}	

    printf("----- BEGINNING SOLVER ON RANK %u -----\n", gfs->domain.rank);
    for (int _=0;;_+=check_every)
    {
    for (int iter = 0; iter < check_every; iter++)
	{
    	exchange_ghost_cells(gfs);
    	for (long j = ystart; j < yend; j++) 
    	{
    	    for (long i = xstart; i < xend; i++)
    	    {
    	    const int ij = gf_indx_2d(gfs, i, j);

		    double lpt; double rpt; double dpt; double upt; 

		    // vvv BOUNDARY CONDITIONS vvv  vvvvvvvvvvvvv REGULAR POINTS vvvvvvvvvvvvv
		    if (i == 0) 	    {lpt = 0;}  else {lpt = uval[gf_indx_2d(gfs, i-1, j)];}
		    if (i == gfs->nx-1) {rpt = 0;}  else {rpt = uval[gf_indx_2d(gfs, i+1, j)];}
		    if (j == 0) 	    {dpt = 0;}  else {dpt = uval[gf_indx_2d(gfs, i, j-1)];}
		    if (j == gfs->ny-1) {upt = 0;}  else {upt = uval[gf_indx_2d(gfs, i, j+1)];}
		    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^ 	^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		    printf("Before: uval[%d] = %f\n", ij, uval[ij]);
    	    uval[ij] = (lpt + rpt + dpt + upt) / 4 - sval[ij] * gfs->dx * gfs->dy;
		    printf("After: uval[%d] = %f\n", ij, uval[ij]);
    	    }
    	}
    }

        double gerror = check_error(gfs);

	if (gfs->domain.rank == 0)
	{
	    printf("%e\n", gerror);
	}

	if (gerror < 1.0e-9)
	{
	    printf("REACHED\n");
	    break;
	}
    }
}


static void fill_guess_values(struct ngfs_2d *gfs) {
    for (int j = 0; j < gfs->ny; j++)
    {
	    const double y = gfs->y0 + j * gfs->dy;
        for (int i = 0; i < gfs->nx; i++)
        {
            const int ij = gf_indx_2d(gfs, i, j);
	        const double x = gfs->x0 + i * gfs->dx;
	        gfs->vars[0]->val[ij] = 1.0;
            //gfs->vars[1]->val[ij] = -2*(M_PI*M_PI)*sin(M_PI*x)*sin(M_PI*y);
            gfs->vars[1]->val[ij] = cos(20 * 4 * atan(1) * x) * cos(20 * 4 * atan(1) * y);
	        gfs->vars[2]->val[ij] = 0.0;
        }
    }	
}


int MPI_Isendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                  MPI_Request *request)
{
    MPI_Isend(sendbuf, sendcount, sendtype, dest, sendtag, comm, request);
    MPI_Irecv(recvbuf, recvcount, recvtype, source, recvtag, comm, request);

    return 0;
}



static void exchange_ghost_cells(struct ngfs_2d *gfs)
{
    int left_rank = gfs->domain.lower_x_rank;
    int right_rank = gfs->domain.upper_x_rank;
    int bottom_rank = gfs->domain.lower_y_rank;
    int top_rank = gfs->domain.upper_y_rank;
    //int rank = gfs->domain.rank;

    MPI_Request request;
    int sstart;
    int rstart;

    for (int j = 0; j < gfs->ny; j++)
    {
        // Left to Right
        if (gfs->domain.upper_x_rank != INVALID_RANK)
        {
            if (gfs->domain.lower_x_rank == INVALID_RANK) {
                // grabs only left edge
                sstart = gf_indx_2d(gfs, gfs->nx - 2 * gfs->gs, j);
                rstart = gf_indx_2d(gfs, gfs->nx - gfs->gs, j);
            }
            else {
                // only cores in the middle
                sstart = gf_indx_2d(gfs, gfs->nx - 2 * gfs->gs, j);
                rstart = gf_indx_2d(gfs, 0, j);
            }
            MPI_Isendrecv(&gfs->vars[0]->val[sstart], gfs->gs, MPI_DOUBLE, right_rank, 0,
                &gfs->vars[0]->val[rstart], gfs->gs, MPI_DOUBLE, left_rank, 0,
                MPI_COMM_WORLD, &request);
        }

        // Right to Left
        if (gfs->domain.lower_x_rank != INVALID_RANK)
        {
            if (gfs->domain.upper_x_rank == INVALID_RANK) {
                // grabs good data from right edge
                sstart = gf_indx_2d(gfs, gfs->gs, j);
                rstart = gf_indx_2d(gfs, 0, j);
            }
            else {
                // only cores in the middle
                sstart = gf_indx_2d(gfs, gfs->gs, j);
                rstart = gf_indx_2d(gfs, gfs->nx - gfs->gs, j);
            }
            MPI_Isendrecv(&gfs->vars[0]->val[sstart], gfs->gs, MPI_DOUBLE, left_rank, 0,
                &gfs->vars[0]->val[rstart], gfs->gs, MPI_DOUBLE, right_rank, 0,
                MPI_COMM_WORLD, &request);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Top to Bottom
    if (gfs->domain.lower_y_rank != INVALID_RANK)
    {
            if (gfs->domain.upper_y_rank == INVALID_RANK) {
                // Top edge sending
                sstart = gf_indx_2d(gfs, 0, gfs->gs); // gs is right
                rstart = gf_indx_2d(gfs, 0, 0); // 0,0 is right
            }
            else {
                // Middle sending
                sstart = gf_indx_2d(gfs, 0, gfs->gs); // gs is right
                rstart = gf_indx_2d(gfs, 0, gfs->ny -gfs->gs); // ny-gs is right
            }

            MPI_Isendrecv(&gfs->vars[0]->val[sstart], gfs->nx * gfs->gs, MPI_DOUBLE, bottom_rank, 0,
                      &gfs->vars[0]->val[rstart], gfs->nx * gfs->gs, MPI_DOUBLE, top_rank, 0,
                      MPI_COMM_WORLD, &request);
    }

    // Bottom to Top
    if (gfs->domain.upper_y_rank != INVALID_RANK)
    {
            if (gfs->domain.lower_y_rank == INVALID_RANK) {
            // Working on bottom edge
                    sstart = gf_indx_2d(gfs, 0, gfs->ny - 2 * gfs->gs);
                    rstart = gf_indx_2d(gfs, 0 , gfs->ny - gfs->gs);
            //printf("This should be the bottom: rank %d \n", rank); // verification is good!
            }
            else {
            // BOTTOM SENDING TO TOP; works on all middle cores
                sstart = gf_indx_2d(gfs, 0, gfs->ny - 2 * gfs->gs);
                rstart = gf_indx_2d(gfs, 0, 0);
            //printf("This should be the middle: rank %d \n", rank);
            }

            MPI_Isendrecv(&gfs->vars[0]->val[sstart], gfs->nx * gfs->gs, MPI_DOUBLE, top_rank, 0,
                      &gfs->vars[0]->val[rstart], gfs->nx * gfs->gs, MPI_DOUBLE, bottom_rank, 0,
                      MPI_COMM_WORLD, &request);
    }
}


double check_error(struct ngfs_2d *gfs)
{
    double error = 0.0;
    double gerror = 0.0;

    int left_rank = gfs->domain.lower_x_rank;
    int right_rank = gfs->domain.upper_x_rank;
    int bottom_rank = gfs->domain.lower_y_rank;
    int top_rank = gfs->domain.upper_y_rank;

    double *uval = gfs->vars[0]->val;
    double *sval = gfs->vars[1]->val;
    double *eval = gfs->vars[2]->val;

    int ystart; int yend; int xstart; int xend;

    if (top_rank == INVALID_RANK)    {yend   = gfs->ny;} else {yend   = gfs->ny-1;}
    if (bottom_rank == INVALID_RANK) {ystart = 0;}       else {ystart = 1;}
    if (left_rank == INVALID_RANK)   {xstart = 0;}       else {xstart = 1;}
    if (right_rank == INVALID_RANK)  {xend   = gfs->nx;} else {xend   = gfs->nx-1;}

    for (long j = ystart; j < yend; j++)
    {
	for (long i = xstart; i < xend; i++)
	{
	    const int ij = gf_indx_2d(gfs, i, j);

	    double lpt; double rpt; double dpt; double upt;

	        if (i == 0) 	    {lpt = 0;}  else {lpt = uval[gf_indx_2d(gfs, i-1, j)];}
            if (i == gfs->nx-1) {rpt = 0;}  else {rpt = uval[gf_indx_2d(gfs, i+1, j)];}
            if (j == 0) 	    {dpt = 0;}  else {dpt = uval[gf_indx_2d(gfs, i, j-1)];}
            if (j == gfs->ny-1) {upt = 0;}  else {upt = uval[gf_indx_2d(gfs, i, j+1)];}

	    eval[ij] = (rpt + lpt + upt + dpt - 4 * uval[ij]) / (gfs->dx * gfs->dy) - sval[ij];
            double lerr = fabs(eval[ij]);
	    if (lerr > error)
	    {
		error = lerr;
	    }
    }
    }


    MPI_Allreduce(&error, &gerror, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return gerror;
}


static void print_gf_function(struct ngfs_2d *gfs)
{
    char fname[128];
    snprintf(fname, 127, "rank_%d.json", gfs->domain.rank);
    FILE *f = fopen(fname, "w");
    fprintf(f, "{\n");
    fprintf(f, "    \"nx\": %lu,\n", gfs->nx);
    fprintf(f, "    \"ny\": %lu,\n", gfs->ny);
    fprintf(f, "    \"dx\": %20.16e,\n", gfs->dx);
    fprintf(f, "    \"dy\": %20.16e,\n", gfs->dy);
    fprintf(f, "    \"x0\": %20.16e,\n", gfs->x0);
    fprintf(f, "    \"y0\": %20.16e,\n", gfs->y0);
    fprintf(f, "    \"data\": [ ");
    for (int j = 0; j < gfs->ny; j++)
    {
        const char *end = (j != gfs->ny - 1) ? "," : "";
        fprintf(f, "[\n");

        for (int i = 0; i < gfs->nx; i++)
        {
            const char *end = i != gfs->nx - 1 ? "," : "";
            const int ij = gf_indx_2d(gfs, i, j);
            fprintf(f, "%20.16e%s", gfs->vars[0]->val[ij], end);
        }
        fprintf(f, "]%s", end);
    }
    fprintf(f, "]");
    fprintf(f, "}\n");
    fclose(f);
}
