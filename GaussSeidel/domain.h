#ifndef DOMAIN_H
#define DOMAIN_H
#include <stdlib.h>

#define INVALID_RANK -1
struct bbox_st
{
    int lower;
    int upper;
};

struct domain1d_st
{
    int rank;
    struct bbox_st bbox;
    int n;
    int lower_size;
    int lower_rank;
    int upper_size;
    int upper_rank;
    int local0;
    int gs;
};

struct domain2d_st
{
    int rank;         /* this processes rank */
    int mpi_size;     /* Total number of mpi processes */
    int gs;           /* number of ghost zones */
    int local_nx;     /* number of points in x direction of local grid */
    int local_ny;     /* number of points in y direction of local grid */
    int local_i0;     /* global grid i index of the local grid i=0 */
    int local_j0;     /* global grid j index of the local grid j=0 */
    int lower_x_rank; /* rank of lower x neighboor */
    int upper_x_rank; /* rank of upper x neighboor */
    int lower_y_rank; /* rank of lower y neighboor */
    int upper_y_rank; /* rank of upper y neighboor */
    size_t global_ni; /* dimensions of global grid */
    size_t global_nj; /* dimensions of global grid */
    double global_x0;  /* global minimum x */
    double global_y0;  /* global minimum y */
    double dx;        /* grid spacing */
    double dy;        /* grid spacing */
};

struct domain3d_st
{
    int rank;         /* this processes rank */
    int mpi_size;     /* Total number of mpi processes */
    int gs;           /* number of ghost zones */
    int local_nx;     /* number of points in x direction of local grid */
    int local_ny;     /* number of points in y direction of local grid */
    int local_nz;     /* number of points in z direction of local grid */
    int local_i0;     /* global grid i index of the local grid i=0 */
    int local_j0;     /* global grid j index of the local grid j=0 */
    int local_k0;     /* global grid k index of the local grid k=0 */
    int lower_x_rank; /* rank of lower x neighboor */
    int upper_x_rank; /* rank of upper x neighboor */
    int lower_y_rank; /* rank of lower y neighboor */
    int upper_y_rank; /* rank of upper y neighboor */
    int lower_z_rank; /* rank of lower z neighboor */
    int upper_z_rank; /* rank of upper z neighboor */
    size_t global_ni; /* dimensions of global grid */
    size_t global_nj; /* dimensions of global grid */
    size_t global_nk; /* dimensions of global grid */
    double global_x0;  /* global minimum x */
    double global_y0;  /* global minimum y */
    double global_z0;  /* global minimum z */
    double dx;        /* grid spacing */
    double dy;        /* grid spacing */
    double dz;        /* grid spacing */
};

int setup_1d_domain(const int ncpu_per_direction, const int direction_rank,
                    const int nglobal, const int gs,
                    struct domain1d_st *domain);

int setup_2d_domain(const int nx_cpu, const int ny_cpu, const int rank,
                    const int global_nx, const int global_ny, const int gs,
                    const double global_x0, const double global_y0,
                    const double dx, const double dy,
                    struct domain2d_st *domain);

int setup_3d_domain(const int nx_cpu, const int ny_cpu, const int nz_cpu,
                    const int rank, const int global_nx, const int global_ny,
                    const int global_nz, const int gs, const double global_x0,
                    const double global_y0, const double global_z0,
                    const double dx, const double dy, const double dz,
                    struct domain3d_st *domain);

#endif
