#include "domain.h"
#include "gf.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_gf_test_function(struct ngfs_2d *gfs);
static void print_gf_test_function(struct ngfs_2d *gfs);
static void exchange_ghost_cells(struct ngfs_2d *gfs);
int MPI_Isendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, 
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                  MPI_Request *request);

int main(int argc, char **argv)
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

    const int gs = 2;

    struct ngfs_2d gfs;
    gfs.vars = NULL;

    const int nvars = 2;

    setup_2d_domain(px, py, mpi_rank, global_nx, global_ny, gs, global_x0,
                    global_y0, dx, dy, &gfs.domain);

    ngfs_2d_allocate(nvars, &gfs);

    // Blocks until all processes complete domain decomposition
    MPI_Barrier(MPI_COMM_WORLD);

    fill_gf_test_function(&gfs);
    exchange_ghost_cells(&gfs);

    // Blocks until all processes swap ghost zones
    MPI_Barrier(MPI_COMM_WORLD);

    print_gf_test_function(&gfs);

    /*
    char fname[128];
    snprintf(fname, 127, "rank_%d", gfs.domain.rank);
    FILE *f = fopen(fname, "w");
    fprintf(f, "lower_x_rank: %d, upper_x_rank: %d\n", gfs.domain.lower_x_rank,
            gfs.domain.upper_x_rank);

    for (int i = 0; i < gfs.nx; i++)
    {
        fprintf(f, "%e\n", gfs.x0 + i * gfs.dx);
    }

    fprintf(f, "lower_y_rank: %d, upper_y_rank: %d\n", gfs.domain.lower_y_rank,
            gfs.domain.upper_y_rank);

    for (int j = 0; j < gfs.ny; j++)
    {
        fprintf(f, "%e\n", gfs.y0 + j * gfs.dy);
    }
    fclose(f);
    */
    MPI_Finalize();
    return EXIT_SUCCESS;
}

static void fill_gf_test_function(struct ngfs_2d *gfs)
{
    for (int j = 0; j < gfs->ny; j++)
    {
        const double y = gfs->y0 + j * gfs->dy;
        for (int i = 0; i < gfs->nx; i++)
        {
            const int ij = gf_indx_2d(gfs, i, j);
            const double x = gfs->x0 + i * gfs->dx;
            gfs->vars[0]->val[ij] = sin(x * 10) * cos(y * 12);
        }
    }

    if (gfs->domain.lower_x_rank != INVALID_RANK)
    {
	for (int j = 0; j < gfs->ny; j++)
	{
	    for (int i = 0; i < gfs->gs; i++)
	    {
		const int ij = gf_indx_2d(gfs, i, j);
		gfs->vars[0]->val[ij] = 2;
	    }
	}
    }

    if (gfs->domain.upper_x_rank != INVALID_RANK)
    {
        for (int j = 0; j < gfs->ny; j++)
        {
            for (int i = gfs->nx - gfs->gs; i < gfs->nx; i++)
            {
                const int ij = gf_indx_2d(gfs, i, j);
                gfs->vars[0]->val[ij] = 2;
            }
        }
    }

    if (gfs->domain.upper_y_rank != INVALID_RANK)
    {
        for (int j = gfs->ny - gfs->gs; j < gfs->ny; j++)
        {
            for (int i = 0; i < gfs->nx; i++)
            {
                const int ij = gf_indx_2d(gfs, i, j);
                gfs->vars[0]->val[ij] = 2;
            }
        }
    }

    if (gfs->domain.lower_y_rank != INVALID_RANK)
    {
        for (int j = 0; j < gfs->gs; j++)
        {
            for (int i = 0; i < gfs->nx; i++)
            {
                const int ij = gf_indx_2d(gfs, i, j);
                gfs->vars[0]->val[ij] = 2;
            }
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

    /*int *b = malloc(gfs->gs * gfs->ny * sizeof(double));
    
    for (i=0; i < 2 * gfs->gs * gfs->ny; i++)
    {
    	const int start = gf_indx_2d(gfs, gfs->nx - 2*gfs->gs - 1, 0);
    	b[i] = gfs->vars[0]->val[start + i];
    }*/
    MPI_Request request;
    int sstart;
    int rstart;
    for (int j=0; j < gfs->ny; j++)
    {
	// Left to Right
	if (gfs->domain.upper_x_rank != INVALID_RANK)
	{
	    sstart = gf_indx_2d(gfs, gfs->nx - 2*gfs->gs, j);
            rstart = gf_indx_2d(gfs, gfs->nx - gfs->gs, j);
            MPI_Isendrecv(&gfs->vars[0]->val[sstart], gfs->gs, MPI_DOUBLE, right_rank, 0, 
    		          &gfs->vars[0]->val[rstart], gfs->gs, MPI_DOUBLE, left_rank, 0,
    		          MPI_COMM_WORLD, &request);
	}
	
	// Right to Left
	if (gfs->domain.lower_x_rank != INVALID_RANK)
	{
	    sstart = gf_indx_2d(gfs, gfs->gs, j);
            rstart = gf_indx_2d(gfs, 0, j);
	    MPI_Isendrecv(&gfs->vars[0]->val[sstart], gfs->gs, MPI_DOUBLE, left_rank, 0,
			  &gfs->vars[0]->val[rstart], gfs->gs, MPI_DOUBLE, right_rank, 0,
			  MPI_COMM_WORLD, &request);
	}
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Top to Bottom
    if (gfs->domain.lower_y_rank != INVALID_RANK)
    {
	start = gf_indx_2d(gfs, 0, gfs->ny - 2*gfs->gs);
	rstart = gf_indx_2d(gfs, 0, 0);

	MPI_Isendrecv(&gfs->vars[0]->val[sstart], gfs->nx * gfs->gs, MPI_DOUBLE, bottom_rank, 0,
		      &gfs->vars[0]->val[rstart], gfs->nx * gfs->gs, MPI_DOUBLE, top_rank, 0,
		      MPI_COMM_WORLD, &request);
    }

    // Bottom to Top
    if (gfs->domain.upper_y_rank != INVALID_RANK)
    {
	sstart = gf_indx_2d(gfs, 0, gfs->gs);
	rstart = gf_indx_2d(gfs, 0, gfs->ny - gfs->gs);

	MPI_Isendrecv(&gfs->vars[0]->val[sstart], gfs->nx * gfs->gs, MPI_DOUBLE, top_rank, 0,
		      &gfs->vars[0]->val[rstart], gfs->nx * gfs->gs, MPI_DOUBLE, bottom_rank, 0,
		      MPI_COMM_WORLD, &request);
    }
}


static void print_gf_test_function(struct ngfs_2d *gfs)
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
