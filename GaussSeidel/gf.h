#ifndef GF_H
#define GF_H

#include "domain.h"
#include <stdlib.h>

/* struct gf holds the data for a single gridfunction */
struct gf
{
    size_t n;             /* length of each array */
    int gs;               /* Ghost size of algorithm */
    double *restrict val; /* array to store data ̑̑*/
    char *vname;          /* A name could be helpful for IO */
};

/* struct ngfs holds data for all gridfunctions */
struct ngfs_3d
{
    int nvars;                 /* How many variables */
    double x0;                 /* local x coordinate "origin" */
    double y0;                 /* local y coordinate "origin" */
    double z0;                 /* local z coordinate "origin" */
    double dx;                 /* x coordinate grid spacing */
    double dy;                 /* y coordinate grid spacing */
    double dz;                 /* z coordinate grid spacing */
    size_t n;                  /* length of arrays */
    size_t nx;                 /* length of arrays */
    size_t ny;                 /* length of arrays */
    size_t nz;                 /* length of arrays */
    int gs;                    /* ghost size */
    struct gf **vars;          /* pointer to nvars gf structures */
    struct domain3d_st domain; /* Domain structure */
};

struct ngfs_2d
{
    int nvars;                 /* How many variables */
    double x0;                 /* local x coordinate "origin" */
    double y0;                 /* local y coordinate "origin" */
    double dx;                 /* x coordinate grid spacing */
    double dy;                 /* y coordinate grid spacing */
    size_t n;                  /* length of arrays */
    size_t nx;                 /* length of arrays */
    size_t ny;                 /* length of arrays */
    int gs;                    /* ghost size */
    struct gf **vars;          /* pointer to nvars gf structures */
    struct domain2d_st domain; /* Domain structure */
};

static inline long gf_indx_2d(struct ngfs_2d *gfs, long i, long j)
{
    return i + j * gfs->nx;
}

static inline int gf_indx_3d(struct ngfs_2d *gfs, int i, int j, int k)
{
    return i + (j + k * gfs->ny) * gfs->nx;
}

int gf_allocate(size_t n, int gs, struct gf *gptr, char *vname);
int gf_deallocate(struct gf *gptr);

int ngfs_3d_allocate(int nvars, struct ngfs_3d *ptr);
int ngfs_2d_allocate(int nvars, struct ngfs_2d *ptr);

int ngfs_3d_deallocate(struct ngfs_3d *ptr);
int ngfs_2d_deallocate(struct ngfs_2d *ptr);

int gf_rename(struct gf *gptr, const char *name);
#endif
