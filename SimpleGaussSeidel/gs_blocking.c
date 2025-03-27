#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const int gz = 1;
const int check_error_every = 1024;

struct comm_st
{
    int mpi_size;            /* total number of processes */
    int mpi_rank;            /* rank of this process */
    int lower_neighbor;      /* rank of lower neighbor */
    int upper_neighbor;      /* rank of upper neighbor */
    long npoints;            /* number of points on local grid */
    long global_i_0;         /* global index of first local point */
};

struct gh_st
{
    double *xval;           /* x - coordinate. */
    double *uval;           /* solution function */
    double *sval;           /* source function */
    double *eval;           /* backwards error */
    double h;               /* grid spacing */
    long npoints;           /* number of points on local grid */
};

void sync(struct gh_st *gh, const struct comm_st *comm);
double check_error(struct gh_st *gh, struct comm_st *comm);
void setup_parallel(struct comm_st *comm, const long ncells);
void output_data(const struct gh_st *gh, const struct comm_st *comm);
void setup_grid_functions(struct gh_st *gh, const struct comm_st *comm,
                          const long ncells);
void cleanup_grid_functions(struct gh_st *gh);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc != 2)
    {
        fprintf(stderr, "Expected one integer argument\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    const long global_ncells = atol(argv[1]);

    if (global_ncells <= 2 * mpi_size)
    {
        fprintf(stderr, "Expected argument n > 2 * num_ranks\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    struct comm_st comm;
    struct gh_st gh;

    setup_parallel(&comm, global_ncells);

    const double omega = 1.9;

    setup_grid_functions(&gh, &comm, global_ncells);

    const long npoints = comm.npoints;

    double *uval = gh.uval;
    double *sval = gh.sval;

    const int global_i_0_is_odd = comm.global_i_0 % 2 == 0;

    const double t_start = MPI_Wtime();

    const double h2 = gh.h * gh.h;
    for (;;)
    {

        for (int it = 0; it < check_error_every; it++)
        {
            /* loop over odd points (on global grid) */
            for (long i = global_i_0_is_odd + 1; i < npoints - 1; i += 2)
            {
                uval[i] =
                    (1.0 - omega) * uval[i] +
                    omega * 0.5 * (uval[i + 1] + uval[i - 1] - h2 * sval[i]);
            }
            sync(&gh, &comm);

            /* loop over even points (on global grid) */
            for (long i = 2 - global_i_0_is_odd; i < npoints - 1; i += 2)
            {
                uval[i] =
                    (1.0 - omega) * uval[i] +
                    omega * 0.5 * (uval[i + 1] + uval[i - 1] - h2 * sval[i]);
            }
            sync(&gh, &comm);
        }

        double gerror = check_error(&gh, &comm);

        if (comm.mpi_rank == 0)
        {
            printf("%e\n", gerror);
        }

        if (gerror < 1.0e-9)
        {
            break;
        }
    }
    const double t_end = MPI_Wtime();

    if (comm.mpi_rank == 0 )
    {
        printf("Total time: %e\n", t_end - t_start);
    }
    output_data(&gh, &comm);

    cleanup_grid_functions(&gh);

    MPI_Finalize();
    return 0;
}

void sync(struct gh_st *gh, const struct comm_st *comm)
{
    double *uval = gh->uval;
    if (comm->lower_neighbor > -1)
    {
        double buffer = uval[1];
        MPI_Send(&buffer, 1, MPI_DOUBLE, comm->lower_neighbor, 0,
                 MPI_COMM_WORLD);
    }
    if (comm->upper_neighbor < comm->mpi_size)
    {
        double buffer = uval[comm->npoints - 2];
        MPI_Send(&buffer, 1, MPI_DOUBLE, comm->upper_neighbor, 0,
                 MPI_COMM_WORLD);
    }
    if (comm->lower_neighbor > -1)
    {
        MPI_Recv(uval, 1, MPI_DOUBLE, comm->lower_neighbor, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    if (comm->upper_neighbor < comm->mpi_size)
    {
        MPI_Recv(uval + comm->npoints - 1, 1, MPI_DOUBLE, comm->upper_neighbor,
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

double check_error(struct gh_st *gh, struct comm_st *comm)
{
    double error = 0.0;
    double gerror = 0.0;
    const double h = gh->h;
    const double h2 = h * h;
    double *eval = gh->eval;
    double *uval = gh->uval;
    double *sval = gh->sval;

    for (long i = 1; i < comm->npoints - 1; i++)
    {
        eval[i] = (uval[i + 1] + uval[i - 1] - 2 * uval[i]) / h2 - sval[i];
        double lerr = fabs(eval[i]);
        if (lerr > error)
        {
            error = lerr;
        }
    }
    MPI_Allreduce(&error, &gerror, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return gerror;
}

void setup_parallel(struct comm_st *comm, const long ncells)
{
    MPI_Comm_size(MPI_COMM_WORLD, &comm->mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm->mpi_rank);

    comm->lower_neighbor = comm->mpi_rank - 1;
    comm->upper_neighbor = comm->mpi_rank + 1;

    int lower_ghost = 0;
    int upper_ghost = 0;
    if (comm->lower_neighbor > -1)
    {
        lower_ghost = gz;
    }
    if (comm->upper_neighbor < comm->mpi_size)
    {
        upper_ghost = gz;
    }

    const long npoints_per_process = (ncells + 1) / comm->mpi_size;
    comm->global_i_0 = npoints_per_process * comm->mpi_rank - lower_ghost;

    comm->npoints = npoints_per_process + lower_ghost + upper_ghost;

    if (comm->mpi_rank == comm->mpi_size - 1)
    {
        comm->npoints =
            (ncells + 1) - npoints_per_process * comm->mpi_rank + lower_ghost;
    }
}

void output_data(const struct gh_st *gh, const struct comm_st *comm)
{
    char name[1000];
    snprintf(name, 1000, "rank_%d.asc", comm->mpi_rank);
    FILE *file = fopen(name, "w");
    for (long i = 0; i < comm->npoints; i++)
    {
        fprintf(file, "%20.16e %20.16e %20.16e\n", gh->xval[i], gh->uval[i],
                gh->eval[i]);
    }
    fclose(file);
}

void setup_grid_functions(struct gh_st *gh, const struct comm_st *comm,
                          const long ncells)
{
    const double h = 1.0 / ncells;

    const long npoints = comm->npoints;

    double *uval = malloc(sizeof(double) * npoints);
    double *xval = malloc(sizeof(double) * npoints);
    double *sval = malloc(sizeof(double) * npoints);
    double *eval = malloc(sizeof(double) * npoints);

    assert(uval);
    assert(xval);
    assert(sval);
    assert(eval);

    gh->uval = uval;
    gh->xval = xval;
    gh->sval = sval;
    gh->eval = eval;
    gh->h = h;

    for (long i = 0; i < npoints; i++)
    {
        const double x = (comm->global_i_0 + i) * h;
        xval[i] = x;
        uval[i] = 0.0;
        sval[i] = cos(20 * 4 * atan(1) * x);
        eval[i] = 0.0;
    }

    printf("%d (%e, %e)\n", comm->mpi_rank, xval[0], xval[npoints - 1]);
}

void cleanup_grid_functions(struct gh_st *gh)
{
    free(gh->xval);
    free(gh->uval);
    free(gh->sval);
    free(gh->eval);
}
