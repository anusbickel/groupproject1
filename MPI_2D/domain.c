#include "domain.h"

/* Consider a grid of the form (11 points):
 *   o o o o o o o o o o o
 *   that will be distributed among 3 MPI processes (x = ghostzone)
 *
 *   o o o o x x   (4 non-ghostzones, 6 total points)
 *       x x o o o o x x  (4 non-ghostzones, 8 total points )
 *               x x o o o  (3 non-ghostzones, 5 total points)
 *
 * That is 11/3 = 3, and 11 - 3*3 = 2, meaning two MPI processes
 * need to have an extra point.
 * Proc 0: local0 = 0, localn = 6.
 * Proc 1: local0 = 2, localn = 8.
 * Proc 2: local0 = 7, localn = 5.
 */
int setup_1d_domain(const int ncpu_per_direction, const int direction_rank,
                    const int nglobal, const int gs, struct domain1d_st *domain)
{
    const int rank = direction_rank;
    const int nprocs = ncpu_per_direction;

    domain->rank = rank;
    /* ln starts as the approximate number of non-ghostzones per proces
     */
    int ln = nglobal / nprocs;

    /* Since ln * nprocs < nglobal, there are processes that will need
     * to have more points. The number of such processes is stored in
     * "under"
     */
    const int under = nglobal - ln * nprocs;

    domain->bbox.lower = 1;
    domain->bbox.upper = 1;
    domain->local0 = 0;

    /* Here we try to find the starting index (on the global grid) of
     * the first non-ghostzone on our local grid. We do this by summing
     * up the non-ghostzone points owned by all processes with rank less
     * than our rank. All ranks < under will have an extra point.
     */
    for (int i = 0; i < under && i < rank; i++)
    {
        domain->local0 += (ln + 1);
    }
    for (int i = under; i < rank; i++)
    {
        domain->local0 += ln;
    }

    /* If our rank < under, then we also have an extra point */
    if (rank < under)
    {
        ln++;
    }

    domain->lower_rank = INVALID_RANK;
    domain->upper_rank = INVALID_RANK;
    domain->lower_size = 0;
    domain->upper_size = 0;

    /* add ghostzones except at bdry */
    if (domain->local0 > 0)
    {
        domain->bbox.lower = 0;
        domain->lower_rank = rank - 1;
        domain->local0 -= gs;
        domain->lower_size = gs;
        ln += gs;
    }
    else
    {
        domain->lower_rank = INVALID_RANK;
    }
    if (domain->local0 + ln < nglobal)
    {
        domain->bbox.upper = 0;
        domain->upper_rank = rank + 1;
        ln += gs;
        domain->upper_size = gs;
    }
    else
    {
        domain->upper_rank = INVALID_RANK;
    }

    /* At this point, ln now contains the number of points, including
     * all ghost zones and local0 contains the index of the first point
     * (which is typically a ghost point).
     */
    domain->n = ln;
    domain->gs = gs;
    return 0;
}

static inline int grid_rank_to_mpi_rank_3d(const int x_rank, const int y_rank,
                                    const int z_rank, const int nx_cpu,
                                    const int ny_cpu)
{
    if (x_rank < 0 || y_rank < 0 || z_rank < 0)
    {
        return INVALID_RANK;
    }
    return x_rank + y_rank * nx_cpu + z_rank * nx_cpu * ny_cpu;
}

static inline int grid_rank_to_mpi_rank_2d(const int x_rank, const int y_rank,
                                    const int nx_cpu)
{
    if (x_rank < 0 || y_rank < 0)
    {
        return INVALID_RANK;
    }
    return x_rank + y_rank * nx_cpu;
}

int setup_3d_domain(const int nx_cpu, const int ny_cpu, const int nz_cpu,
                    const int rank, const int nx_global, const int ny_global,
                    const int nz_global, const int gs, const double global_x0,
                    const double global_y0, const double global_z0,
                    const double dx, const double dy, const double dz,
                    struct domain3d_st *domain)
{
    struct domain1d_st domain_1d;

    domain->global_ni = nx_global;
    domain->global_nj = ny_global;
    domain->global_nk = nz_global;
    domain->gs = gs;

    domain->global_x0 = global_x0;
    domain->global_y0 = global_y0;
    domain->global_z0 = global_z0;

    domain->dx = dx;
    domain->dy = dy;
    domain->dz = dz;

    domain->mpi_size = nx_cpu * ny_cpu * nz_cpu;

    const int rank_z = rank / (nx_cpu * ny_cpu);
    const int rem = rank % (nx_cpu * ny_cpu);
    const int rank_y = rem / nx_cpu;
    const int rank_x = rem % nx_cpu;
    setup_1d_domain(nx_cpu, rank_x, nx_global, gs, &domain_1d);

    domain->rank = rank;
    domain->local_nx = domain_1d.n;
    domain->local_i0 = domain_1d.local0;
    domain->lower_x_rank = grid_rank_to_mpi_rank_3d(
        domain_1d.lower_rank, rank_y, rank_z, nx_cpu, ny_cpu);
    domain->upper_x_rank = grid_rank_to_mpi_rank_3d(
        domain_1d.upper_rank, rank_y, rank_z, nx_cpu, ny_cpu);

    setup_1d_domain(ny_cpu, rank_y, ny_global, gs, &domain_1d);
    domain->local_ny = domain_1d.n;
    domain->local_j0 = domain_1d.local0;
    domain->lower_y_rank = grid_rank_to_mpi_rank_3d(
        rank_x, domain_1d.lower_rank, rank_z, nx_cpu, ny_cpu);
    domain->upper_y_rank = grid_rank_to_mpi_rank_3d(
        rank_x, domain_1d.upper_rank, rank_z, nx_cpu, ny_cpu);

    setup_1d_domain(nz_cpu, rank_z, nz_global, gs, &domain_1d);
    domain->local_nz = domain_1d.n;
    domain->local_k0 = domain_1d.local0;
    domain->lower_z_rank = grid_rank_to_mpi_rank_3d(
        rank_x, rank_y, domain_1d.lower_rank, nx_cpu, ny_cpu);
    domain->upper_z_rank = grid_rank_to_mpi_rank_3d(
        rank_x, rank_y, domain_1d.upper_rank, nx_cpu, ny_cpu);
    return 0;
}

int setup_2d_domain(const int nx_cpu, const int ny_cpu, const int rank,
                    const int nx_global, const int ny_global, const int gs,
                    const double global_x0, const double global_y0,
                    const double dx, const double dy,
                    struct domain2d_st *domain)
{
    struct domain1d_st domain_1d;

    domain->global_ni = nx_global;
    domain->global_nj = ny_global;
    domain->gs = gs;

    domain->global_x0 = global_x0;
    domain->global_y0 = global_y0;

    domain->dx = dx;
    domain->dy = dy;

    domain->mpi_size = nx_cpu * ny_cpu;

    const int rank_y = rank / nx_cpu;
    const int rank_x = rank % nx_cpu;
    setup_1d_domain(nx_cpu, rank_x, nx_global, gs, &domain_1d);

    domain->rank = rank;
    domain->local_nx = domain_1d.n;
    domain->local_i0 = domain_1d.local0;
    domain->lower_x_rank =
        grid_rank_to_mpi_rank_2d(domain_1d.lower_rank, rank_y, nx_cpu);
    domain->upper_x_rank =
        grid_rank_to_mpi_rank_2d(domain_1d.upper_rank, rank_y, nx_cpu);

    setup_1d_domain(ny_cpu, rank_y, ny_global, gs, &domain_1d);
    domain->local_ny = domain_1d.n;
    domain->local_j0 = domain_1d.local0;
    domain->lower_y_rank =
        grid_rank_to_mpi_rank_2d(rank_x, domain_1d.lower_rank, nx_cpu);
    domain->upper_y_rank =
        grid_rank_to_mpi_rank_2d(rank_x, domain_1d.upper_rank, nx_cpu);

    return 0;
}
