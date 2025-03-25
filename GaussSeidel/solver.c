#include "domain.h"
#include "gf.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_guess_values(struct ngfs_2d *gfs);
static void exchange_ghost_cells(struct ngfs_2d *gfs);
static void print_gf_function(struct ngfs_2d *gfs);
int MPI_Isendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                  MPI_Request *request);



static void fill_guess_values(struct ngfs_2d *gfs) {
    for (int j = 0; j < gfs->ny; j++)
    {
        for (int i = 0; i < gfs->nx; i++)
        {
            const int ij = gf_indx_2d(gfs, i, j);
            gfs->vars[0]->val[ij] = 0;
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

    /*int *b = malloc(gfs->gs * gfs->ny * sizeof(double));

    for (i=0; i < 2 * gfs->gs * gfs->ny; i++)
    {
        const int start = gf_indx_2d(gfs, gfs->nx - 2*gfs->gs - 1, 0);
        b[i] = gfs->vars[0]->val[start + i];
    }*/
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
