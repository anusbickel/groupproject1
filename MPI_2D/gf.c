#include "gf.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Allocate gridfunctions. External routines call this function */
int ngfs_3d_allocate(int nvars, struct ngfs_3d *ptr)
{
    assert(ptr->vars == NULL);
    ptr->nvars = nvars;
    ptr->nx = ptr->domain.local_nx;
    ptr->ny = ptr->domain.local_ny;
    ptr->nz = ptr->domain.local_nz;
    ptr->n = ptr->domain.local_nx * ptr->domain.local_ny * ptr->domain.local_nz;
    ptr->gs = ptr->domain.gs;
    ptr->dx = ptr->domain.dx;
    ptr->dy = ptr->domain.dy;
    ptr->dz = ptr->domain.dz;
    ptr->x0 = ptr->domain.global_x0 + ptr->domain.local_i0 * ptr->domain.dx;
    ptr->y0 = ptr->domain.global_y0 + ptr->domain.local_j0 * ptr->domain.dy;
    ptr->z0 = ptr->domain.global_z0 + ptr->domain.local_k0 * ptr->domain.dz;

    ptr->vars = calloc(nvars, sizeof(struct gf *));
    assert(ptr->nvars);

    char *vname = NULL;
    const size_t name_length = 20;
    for (int i = 0; i < nvars; i++)
    {
        vname = calloc(name_length, sizeof(char));
        snprintf(vname, name_length, "Var%d", i);
        ptr->vars[i] = calloc(1, sizeof(struct gf));
        gf_allocate(ptr->nx * ptr->ny * ptr->nz, ptr->gs, ptr->vars[i], vname);
        free(vname);
        vname = NULL;
    }

    return 0;
}

int ngfs_2d_allocate(int nvars, struct ngfs_2d *ptr)
{
    assert(ptr->vars == NULL);
    ptr->nvars = nvars;
    ptr->nx = ptr->domain.local_nx;
    ptr->ny = ptr->domain.local_ny;
    ptr->n = ptr->domain.local_nx * ptr->domain.local_ny;
    ptr->gs = ptr->domain.gs;
    ptr->dx = ptr->domain.dx;
    ptr->dy = ptr->domain.dy;
    ptr->x0 = ptr->domain.global_x0 + ptr->domain.local_i0 * ptr->domain.dx;
    ptr->y0 = ptr->domain.global_y0 + ptr->domain.local_j0 * ptr->domain.dy;

    ptr->vars = calloc(nvars, sizeof(struct gf *));
    assert(ptr->nvars);

    char *vname = NULL;
    const size_t name_length = 20;
    for (int i = 0; i < nvars; i++)
    {
        vname = calloc(name_length, sizeof(char));
        snprintf(vname, name_length, "Var%d", i);
        ptr->vars[i] = calloc(1, sizeof(struct gf));
        gf_allocate(ptr->nx * ptr->ny, ptr->gs, ptr->vars[i], vname);
        free(vname);
        vname = NULL;
    }
    return 0;
}






/* free gridfunctions */
int ngfs_deallocate_3d(struct ngfs_3d *ptr)
{
    assert(ptr->vars);

    for (int i = 0; i < ptr->nvars; i++)
    {
        gf_deallocate(ptr->vars[i]);
        free(ptr->vars[i]);
        ptr->vars[i] = NULL;
    }
    free(ptr->vars);
    ptr->nvars = 0;
    ptr->n = 0;
    ptr->gs = 0;
    ptr->vars = NULL;
    return 0;
}

int ngfs_deallocate_2d(struct ngfs_2d *ptr)
{
    assert(ptr->vars);

    for (int i = 0; i < ptr->nvars; i++)
    {
        gf_deallocate(ptr->vars[i]);
        free(ptr->vars[i]);
        ptr->vars[i] = NULL;
    }
    free(ptr->vars);
    ptr->nvars = 0;
    ptr->n = 0;
    ptr->gs = 0;
    ptr->vars = NULL;
    return 0;
}
/* Allocate individual gridfunction. External routines generally don't call this
 * function */
int gf_allocate(size_t n, int gs, struct gf *gptr, char *vname)
{
    gptr->n = n;
    gptr->gs = gs;
    gptr->val = calloc(n, sizeof(double));
    assert(gptr->val);

    if (vname)
    {
        const size_t slen = strlen(vname) + 1;
        gptr->vname = calloc(slen, sizeof(char));

        assert(gptr->vname);

        strncpy(gptr->vname, vname, slen);
        gptr->vname[slen] = '\0';
    }
    else
    {
        gptr->vname = NULL;
    }

    return 0;
}

int gf_rename(struct gf *gptr, const char *name)
{
    const size_t s_len = strlen(name);

    if (s_len < 1)
    {
        fprintf(stderr, "Invalid variable name\n");
        return -1;
    }

    if (s_len > 512)
    {
        fprintf(stderr, "Invalid variable name\n");
        return -2;
    }

    if (gptr->vname)
    {
        free(gptr->vname);
        gptr->vname = NULL;
    }

    char *newname = calloc(s_len + 1, sizeof(char));
    strncpy(newname, name, s_len + 1);
    gptr->vname = newname;

    return 0;
}

/* deallocate individual gridfunction. External routines generally don't call
 * this function */
int gf_deallocate(struct gf *gptr)
{
    free(gptr->val);
    gptr->val = NULL;

    if (gptr->vname)
    {
        free(gptr->vname);
        gptr->vname = NULL;
    }
    gptr->n = 0;
    gptr->gs = 0;

    return 0;
}
