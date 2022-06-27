/*
 * This software was developed by the Thermal Modeling and Analysis
 * Project(TMAP) of the National Oceanographic and Atmospheric
 * Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
 * hereafter referred to as NOAA/PMEL/TMAP.
 *
 * Access and use of this software shall impose the following
 * obligations and understandings on the user. The user is granteHd the
 * right, without any fee or cost, to use, copy, modify, alter, enhance
 * and distribute this software, and any derivative works thereof, and
 * its supporting documentation for any purpose whatsoever, provided
 * that this entire notice appears in all copies of the software,
 * derivative works and supporting documentation.  Further, the user
 * agrees to credit NOAA/PMEL/TMAP in any publications that result from
 * the use of this software or in any product that includes this
 * software. The names TMAP, NOAA and/or PMEL, however, may not be used
 * in any advertising or publicity to endorse or promote any products
 * or commercial entity unless specific written permission is obtained
 * from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
 * is not obligated to provide the user with any support, consulting,
 * training or assistance of any kind with regard to the use, operation
 * and performance of this software nor to provide the user with any
 * updates, revisions, new versions or "bug fixes".
 *
 * THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
 * INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
 * RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
 * CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
 */

/* NCF_Util.c
 *
 * Ansley Manke
 * Ferret V600 April 26, 2005
 * V5600 *acm* fix declarations of fillc and my_len as required by solaris compiler
 *
 * This file contains all the utility functions which Ferret
 * needs in order to do attribute handling. Based on code for EF's.
 * calls are made to nc_ routines from netcdf library.
 */

/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build */
/* *acm  10/06 v601 - Fix by Remik for bug 1455, altix. For string attributes,  */
/*                    allocate one more than the att.len, presumably for the null  */
/*                    terminator for the string. Also double check the string length */
/*                    that is returned from the call to nc_inq_att, and make sure */
/*                    we allocate the correct amount of memory for the string.*/
/* */
/* *acm  11/06 v601 - ncf_delete_var_att didnt reset the attribute id's.  Fix this. */
/*   */
/* *acm  11/06 v601 - new routine ncf_add_var_num_att_dp */
/* *acm  11/06 v601 - new routine ncf_repl_var_att_dp */
/* *acm  11/06 v601 - in ncf_init_other_dset, set the name of the global attribute  */
/*                    to history, and define its attribute type and outflag.*/
/* *acm  11/06 v601 - new routine ncf_rename_var, for fix of bug 1471 */
/* *acm  11/06 v601 - in ncf_delete_var_att, renumber the attid for the remaining attributes. */
/* *acm* 12/06 v602 - new attribute assigned to coordinate vars on input, orig_file_axname */
/* *acm*  2/07 V602   Fix bug 1492, changing attributes of coordinate variables; use pseudo-dataset */
/*                    of user-defined axes to keep track of attributes. */
/* *acm* 10 07        Patches for memory-leak fixes from Remiz Ziemlinski */
/* *acm* 10/07        Further fixes by Remik, initializing att.vals, att.string to NULL, */
/*                    set var.ndims = 0 in ncf_init_other_dset */
/* *acm*  3/08        Fix bug 1534; needed to initialize attribute output flag for */
/*                    the bounds attribute on coordinate axes.*/
/* *acm*  1/09        If adding a new global attribute, also increment ngatts. */
/* *acm*  1/09        Fix bug 1620; In ncf_add_var, which is used when defining user  */
/*                    variables, and also for reading in EZ datasets, I had the default */
/*                    attribute type for missing_value attribute set to NC_DOUBLE. There's no  */
/*                    reason for this as these variables are always single precision.*/
/* *acm* 5/09 *acm*   Fix bug 1664. For user variables, varid matches the uvar from Ferret. */
/*                    therefore it may be larger than nc_ptr->nvars */
/* *acm* 3/11 *acm*   Fix bug 1825. Routine ncf_get_var_seq no longer called */
/* *acm*  1/12      - Ferret 6.8 ifdef double_p for double-precision ferret, see the */
/*                    definition of macro DFTYPE in ferretmacros.h. */
/* *acm*  5/12        V6.8 Additions for creating aggregate datasets */
/* *acm*  8/13        Fix bug 2089. Mark the scale_factor and add_offset attributes   */
/*                    to-be-output when writing variables. */
/* *acm*  8/13        Fix bug 2091. If a string variable has the same name as a dimension, */
/*                    DO NOT mark it as an axis.
*  *acm*  v694 1/15   For ticket 2227: if a dimension from a nc file is not also a
*                     1-D coordinate var, don't write the axis Ferret creates. Do report
*                     in dimnames outputs the dimension names as used by Ferret e.g. a
*                     renamed axis TIME -> TIME1 */
/* *sh*  12/15        Bug fix: ncf_get_agg_member is called with the sequence number of the
 *                    desired dataset. So store that in order to locate the right member */
/* *acm*  2/16        Additions for ticket 2352: LET/D variables and attributes. User-variables
*                     defined with LET/D=n are stored with dataset n. A flag in the ncvar
*                     structure tells that the variable is a user-var. A new subroutine call,
*                     ncf_get_var_uvflag returns this flag, so that SHOW DATA/ATTRIBUTES can
*                     list these variables. When user variables are canceled, the varids for
*                     user-variables remaining in the dataset are adjusted. */
/* *sh*  5/16         added grid management for uvars -- dset/grid paris stored in a LIST
                      replaced uvflag with uvarid */
/* *acm* 6/16         Make sure var.nmemb is initialized when adding a new variable. */
/* *kms* 8/16         Rework the entire file to remove memory leaks and improve consistent initialization */
/* *acm* 10/16        ncf_get_uvar_grid now returns the datatype as well as the grid  */
/* *acm*  5/18        Write a note if error with attribute type that results in bad_file_attr return   */
/* *acm* 12/18        Issue 1049: Issue a NOTE if a dimension-named variable is non-numeric so can't 
                      be an axis  */
/* *acm*  6/19        Don't write a note about string-valued coordinate variables  */
/* *acm*  6/22        issue 112: detect and report dimension too large for 4-byte integer indexing  */

#include <Python.h> /* make sure Python.h is first */
#include <stddef.h>             /* size_t, ptrdiff_t; gfortran on linux rh5*/
#include <unistd.h>             /* for convenience */
#include <stdlib.h>             /* for convenience */
#include <stdio.h>              /* for convenience */
#include <string.h>             /* for convenience */
#include <fcntl.h>              /* for fcntl() */
#include <assert.h>
#include <sys/types.h>          /* required for some of our prototypes */
#include <sys/stat.h>
#include <sys/errno.h>
#include <netcdf.h>

#include "fmtprotos.h"
#include "list.h"               /* locally added list library */
#include "NCF_Util.h"
#include "FerMem.h"

/* List of all datasets (static, so only visible locally) */
static LIST *GLOBAL_ncdsetList = NULL;

/* .... Functions only called internally (thus static) .... */
static ncdset *ncf_get_ds_ptr(int *);
static LIST *ncf_get_ds_varlist(int *);
static ncvar *ncf_get_ds_var_ptr(int *, int *);
static void ncf_init_attribute(ncatt *att_ptr);
static void ncf_free_attribute(char *);
static void ncf_free_attlist(ncvar *);
static void ncf_init_variable(ncvar *);
static void ncf_free_variable(char *);
static void ncf_init_dataset(ncdset *);
static void ncf_free_dataset(char *);

static int initialize_output_flag (char *, int);
static int NCF_ListTraverse_FoundDsetName( char *, char * );
static int NCF_ListTraverse_FoundDsetID( char *, char * );
static int NCF_ListTraverse_FoundVarName( char *, char * );
static int NCF_ListTraverse_FoundVarNameCase( char *, char * );
static int NCF_ListTraverse_FoundVarID( char *, char * );
static int NCF_ListTraverse_FoundUvarID( char *, char * );
static int NCF_ListTraverse_FoundVarAttName( char *, char * );
static int NCF_ListTraverse_FoundVarAttNameCase( char *, char * );
static int NCF_ListTraverse_FoundVarAttID( char *, char * );
static int NCF_ListTraverse_FoundVariMemb( char *, char * );
static int NCF_ListTraverse_FoundDsMemb( char *, char * );
static int NCF_ListTraverse_FoundDsMemb( char *, char * );
static int NCF_ListTraverse_FoundGridDset( char *, char * );

/* ----
 * Find a dataset based on its integer ID and return the scalar information:
 * ndims, nvars, ngatts, recdim.
 */
int FORTRAN(ncf_inq_ds)( int *dset, int *ndims, int *nvars, int *ngatts, int *recdim )
{
    ncdset *nc_ptr;

    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    *ndims = nc_ptr->ndims;
    *nvars = nc_ptr->nvars;
    *ngatts = nc_ptr->ngatts;

    /* dimension for Fortran, add 1 */
    *recdim = nc_ptr->recdim+1;

    return FERR_OK;
}

/* ----
 * Find a dataset based on its integer ID and return the dimension info for
 * dimension given.
 */
int FORTRAN(ncf_inq_ds_dims)( int *dset, int *idim, char dname[], int *namelen, int *dimsize)
{
    ncdset *nc_ptr;

    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    strcpy (dname, nc_ptr->dims[*idim-1].name);
    *namelen = strlen(dname);
    *dimsize = nc_ptr->dims[*idim-1].size;

    return FERR_OK;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and
 * variable id. Return the variable name (in its original upper/lower
     case form), type, nvdims, vdims, nvatts.
 */
int FORTRAN(ncf_inq_var)( int *dset, int *varid, char newvarname[], int *len_newvarname, int *vtype,
                          int *nvdims, int *nvatts, int* coord_var, int *outflag, int *vdims )
{
    ncvar *var_ptr;
    int i;

    /*
     * Get the list of variables.
     */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    strcpy(newvarname, var_ptr->name);
    *len_newvarname = strlen(newvarname);
    *vtype = var_ptr->type;
    *nvdims = var_ptr->ndims;
    *nvatts = var_ptr->natts;
    *outflag = var_ptr->all_outflag;
    *coord_var = var_ptr->is_axis;

    for (i = 0; i < var_ptr->ndims; i++) {
        vdims[i] = var_ptr->dims[i];
    }

    return FERR_OK;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and
 * variable id. Return the variable output type.
 */
int FORTRAN(ncf_get_var_outtype)( int *dset, int *varid,    int *outtype )
{
    ncvar *var_ptr;

    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    *outtype = var_ptr->outtype;
    return FERR_OK;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and
 * variable id. Return the variable type.
 */
int FORTRAN(ncf_get_var_type)( int *dset, int *varid, int *vartype )
{
    ncvar *var_ptr;

    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    *vartype = var_ptr->type;
    return FERR_OK;
}

/* ----
 * Find a variable attribute based on its variable ID and dataset ID, and attribute name
 * Return the attribute name, type, length, and output flag
 */
int FORTRAN(ncf_inq_var_att)( int *dset, int *varid, int *attid, char attname[],
                              int *namelen, int *attype, int *attlen, int *attoutflag )
{
    ncatt *att_ptr;
    ncvar *var_ptr;
    int status;
    LIST *varattlist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;

    /*
     * The list of attributes for the variable in the dataset
     */
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
         return ATOM_NOT_FOUND;

    /* Get the attribute from this list */
    status = list_traverse(varattlist, (char *) attid, NCF_ListTraverse_FoundVarAttID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    att_ptr=(ncatt *)list_curr(varattlist);

    strcpy(attname, att_ptr->name);
    *namelen = strlen(attname);
    *attype = att_ptr->type;
    *attlen = att_ptr->len;
    *attoutflag = att_ptr->outflag;

    return FERR_OK;
}

/* ----
 * Find a dataset based on its name and
 * return the ferret dataset number.
 */
int FORTRAN(ncf_get_dsnum)( char name[] )
{
    ncdset *nc_ptr;
    int status;

    /*
     * Find the dataset.
     */
    status = list_traverse(GLOBAL_ncdsetList, name, NCF_ListTraverse_FoundDsetName, (LIST_FRNT | LIST_FORW | LIST_ALTR));

    /*
     * If the search failed, set the dset to ATOM_NOT_FOUND.
     */
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    nc_ptr=(ncdset *)list_curr(GLOBAL_ncdsetList);
    return nc_ptr->fer_dsetnum;
}

/* ----
 * Find a dataset based on its integer ID and return the name.
 */
int FORTRAN(ncf_get_dsname)( int *dset, char name[] )
{
    ncdset *nc_ptr;

    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    strcpy(name, nc_ptr->fername);
    return FERR_OK;
}

/* ----
 * Find a dataset based on its integer ID and a dimension name. Return the dimension ID.
 */
int FORTRAN(ncf_get_dim_id)( int *dset, char dname[] )
{
    ncdset *nc_ptr;
    int idim;
    int sz;
    int szdim;

    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    for (idim = 0; idim < nc_ptr->ndims; idim++) {
        sz = strlen(dname);
        szdim = strlen(nc_ptr->dims[idim].name);
        if ( (sz == szdim) &&
                 (nc_ptr->dims[idim].size !=0) &&
                 (strncmp(dname, nc_ptr->dims[idim].name, sz) == 0) ) {
            return (idim + 1);
        }
    }

    return ATOM_NOT_FOUND;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and
 * variable id. Return the variable name.
 */
int FORTRAN(ncf_get_var_name)( int *dset, int* ivar, char* string, int* len_name )
{
    ncvar *var_ptr;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, ivar);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    strcpy(string, var_ptr->name);
    *len_name = strlen(string);

    return FERR_OK;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and
 * variable name. Return the variable id, or NOT FOUND if it does not exist
 */
int FORTRAN(ncf_get_var_id)( int *dset, int *varid, char string[] )
{
    ncvar *var_ptr;
    int status;
    LIST *varlist;

    /*
     * Get the list of variables.
     */
    varlist = ncf_get_ds_varlist(dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;
    status = list_traverse(varlist, string, NCF_ListTraverse_FoundVarName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    var_ptr=(ncvar *)list_curr(varlist);
    *varid = var_ptr->varid;

    return FERR_OK;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and
 * variable name. Return the variable id, or NOT FOUND if it does not exist
 */
int FORTRAN(ncf_get_var_id_case)( int *dset, int *varid, char string[] )
{
    ncvar *var_ptr;
    int status;
    LIST *varlist;

    /*
     * Get the list of variables.
     */
    varlist = ncf_get_ds_varlist(dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;
    status = list_traverse(varlist, string, NCF_ListTraverse_FoundVarNameCase, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    var_ptr=(ncvar *)list_curr(varlist);
    *varid = var_ptr->varid;

    return FERR_OK;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and
 * variable ID. Return the coordinate-axis flag.
 */
int FORTRAN(ncf_get_var_axflag)( int *dset, int *varid, int* coord_var, int* ax_dir )
{
    ncvar *var_ptr;

    /*
     * Get the list of variables.
     */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    *coord_var = var_ptr->is_axis;
    *ax_dir = var_ptr->axis_dir;

    return FERR_OK;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and
 * variable ID. Return the variable all_outflag for attributes
 */
int FORTRAN(ncf_get_var_outflag)( int *dset, int *varid, int *iflag )
{
    ncvar *var_ptr;

    /*
     * Get the list of variables and the variable based on its id
     */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    *iflag = var_ptr->all_outflag;

    return FERR_OK;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and
 * variable ID. Return the return the flag indicating file
 * variable vs user-variable
 */
int FORTRAN(ncf_get_var_uvflag)( int *dset, int *varid, int *uvflag )
{
    ncvar *var_ptr;

    /*
     * Get the list of variables and the variable based on its id
     */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    if (var_ptr->uvarid == 0)
        *uvflag = 0;
    else
        *uvflag = 1;

    return FERR_OK;
}


/* ----
 * Find a variable attribute based on the dataset ID and variable ID and attribute name
 * Return the attribute ID
 */
int FORTRAN(ncf_get_var_attr_id)( int *dset, int *varid, char* attname, int* attid )
{
    ncvar *var_ptr;
    ncatt *att_ptr;
    int status;
    LIST *varattlist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * The list of attributes for the variable in the dataset.
     */
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    /* Find attname */
    status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    att_ptr = (ncatt *) list_curr(varattlist);
    *attid = att_ptr->attid;

    return FERR_OK;
}

/* ----
 * Find a variable attribute based on the dataset ID and variable ID and attribute name
 * Return the attribute ID
 */
int FORTRAN(ncf_get_var_attr_id_case)( int *dset, int *varid, char* attname, int* attid )
{
    ncvar *var_ptr;
    ncatt *att_ptr;
    int status;
    LIST *varattlist;

    /*
     * Get the list of variables.
     */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * The list of attributes for the variable in the dataset.
     */
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    /* Find attname */
    status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttNameCase, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    att_ptr = (ncatt *) list_curr(varattlist);
    *attid = att_ptr->attid;

    return FERR_OK;
}

/* ----
 * Find a variable attribute based on the dataset ID and variable ID and attribute ID.
 * Return the attribute name.
 */
int FORTRAN(ncf_get_var_attr_name)( int *dset, int *varid, int* attid, int *namelen, char* attname )
{
    ncvar *var_ptr;
    ncatt *att_ptr;
    int status;
    LIST *varattlist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * The list of attributes for the variable in the dataset
     */
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    status = list_traverse(varattlist, (char *) attid, NCF_ListTraverse_FoundVarAttID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    att_ptr = (ncatt *) list_curr(varattlist);
    strcpy(attname, att_ptr->name);
    *namelen = strlen(attname);

    return FERR_OK;
}

/*----
 * Find a variable attribute based on the dataset ID and variable ID and attribute name.
 * On input, len is the max len to load.
 * Return the attribute, len, and its string or numeric value.
 */
int FORTRAN(ncf_get_var_attr)( int *dset, int *varid, char* attname, char* string, int *len, double* val )
{
    ncvar *var_ptr;
    ncatt *att_ptr;
    int status;
    int i;
    LIST *varattlist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * The list of attributes for the variable in the dataset
     */
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    /* Get the attribute from this list */
    status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    att_ptr = (ncatt *) list_curr(varattlist);
    if ( (att_ptr->type == NC_CHAR) || (att_ptr->type == NC_STRING) ) {
        strncpy(string, att_ptr->string, *len);
        val[0] = NC_FILL_DOUBLE;
    }
    else {
        strcpy(string, "");
        for (i = 0; i < att_ptr->len; i++) {
            val[i] = att_ptr->vals[i];
        }
    }
    *len = att_ptr->len;

    return FERR_OK;
}

/*----
 * Find a numeric attribute based on the dataset ID and variable ID and attribute id.
 * On input, len is the max len to load.
 * Return the attribute, len, and or numeric value.
 */
int FORTRAN(ncf_get_attr_from_id)( int *dset, int *varid, int *attid, int *len, double* val )
{
    ncvar *var_ptr;
    ncatt *att_ptr;
    int status;
    int i;
    LIST *varattlist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * The list of attributes for the variable in the dataset.
     */
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    /* Find the attribute from its ID */
    status = list_traverse(varattlist, (char *) attid, NCF_ListTraverse_FoundVarAttID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    att_ptr = (ncatt *) list_curr(varattlist);

    if ( (att_ptr->type == NC_CHAR) || (att_ptr->type == NC_STRING) ) {
        val[0] = NC_FILL_DOUBLE;
        fprintf(stderr, "ERROR: ncf_get_attr_from_id: Atribute is CHAR or STRING. This function only for numeric.\n");
        return -1;
    }

    for (i = 0; i < att_ptr->len; i++) {
        val[i] = att_ptr->vals[i];
    }
    *len = att_ptr->len;

    return FERR_OK;
}

/* ----
 * Initialize new dataset to contain user variables and
 * save in GLOBAL_ncdsetList for attribute handling
 */
int FORTRAN(ncf_init_uvar_dset)( int *setnum )
{
    ncdset nc;
    ncvar var;            /* variable */
    ncatt att;            /* attribute */

    ncf_init_dataset(&nc);
    strcpy(nc.fername, "UserVariables");
    strcpy(nc.fullpath, " ");
    nc.fer_dsetnum = *setnum;
    nc.ngatts = 1;

    /* set one global attribute, treat as pseudo-variable . the list of variables */
    ncf_init_variable(&var);
    strcpy(var.name, ".");
    var.type = NC_CHAR;
    var.outtype = NC_CHAR;
    var.varid = 0;
    var.natts = 1;
    var.varattlist = list_init(__FILE__, __LINE__);
    if ( var.varattlist == NULL ) {
        fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize GLOBAL attributes list.\n");
        return -1;
    }

    ncf_init_attribute(&att);
    att.outflag = 1;
    att.type = NC_CHAR;
    att.outtype = NC_CHAR;
    att.len = 21;
    strcpy(att.name, "FerretUserVariables" );
    att.string = (char*)FerMem_Malloc(2*sizeof(char), __FILE__, __LINE__);
    strcpy(att.string, " ");

    /* Save attribute in linked list of attributes for variable */
    list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    /* global attributes list complete */

    /*Save variable in linked list of variables for this dataset */
    nc.dsetvarlist = list_init(__FILE__, __LINE__);
    if ( nc.dsetvarlist == NULL ) {
        fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize variable list.\n");
        return -1;
    }
    list_insert_after(nc.dsetvarlist, (char *) &var, sizeof(ncvar), __FILE__, __LINE__);

    /* Add dataset to global nc dataset linked list*/
    if ( GLOBAL_ncdsetList == NULL ) {
        GLOBAL_ncdsetList = list_init(__FILE__, __LINE__);
        if ( GLOBAL_ncdsetList == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize GLOBAL_ncDsetList.\n");
            return -1;
        }
    }
    list_insert_after(GLOBAL_ncdsetList, (char *) &nc, sizeof(ncdset), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Initialize new dataset to contain user-defined coordinate variables and
 * save in GLOBAL_ncdsetList for attribute handling
 */
int FORTRAN(ncf_init_uax_dset)( int *setnum )
{
    ncdset nc;
    ncvar var;            /* variable */
    ncatt att;            /* attribute */

    ncf_init_dataset(&nc);
    strcpy(nc.fername, "UserCoordVariables");
    strcpy(nc.fullpath, " ");
    nc.fer_dsetnum = *setnum;
    nc.ngatts = 1;

    /* set one global attribute, treat as pseudo-variable . the list of variables */

    ncf_init_variable(&var);
    strcpy(var.name, "."); /*is this a legal name?*/
    var.type = NC_CHAR;
    var.outtype = NC_CHAR;
    var.varid = 0;
    var.natts = 1;

    ncf_init_attribute(&att);
    att.outflag = 1;
    att.type = NC_CHAR;
    att.outtype = NC_CHAR;
    att.len = 21;
    strcpy(att.name, "FerretUserCoordVariables" );
    att.string = (char*)FerMem_Malloc(2*sizeof(char), __FILE__, __LINE__);
    strcpy(att.string, " ");

    /* Save attribute in linked list of attributes. */
    var.varattlist = list_init(__FILE__, __LINE__);
    if ( var.varattlist == NULL ) {
        fprintf(stderr, "ERROR: ncf_init_uax_dset: Unable to initialize GLOBAL attributes list.\n");
        return -1;
    }
    list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    /* global attributes list complete */

    /* Save variable in linked list of variables for this dataset */
    nc.dsetvarlist = list_init(__FILE__, __LINE__);
    if ( nc.dsetvarlist == NULL ) {
        fprintf(stderr, "ERROR: ncf_init_uax_dset: Unable to initialize variable list.\n");
        return -1;
    }
    list_insert_after(nc.dsetvarlist, (char *) &var, sizeof(ncvar), __FILE__, __LINE__);

    /* Add dataset to global nc dataset linked list*/
    if ( GLOBAL_ncdsetList == NULL ) {
        GLOBAL_ncdsetList = list_init(__FILE__, __LINE__);
        if ( GLOBAL_ncdsetList == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_uax_dset: Unable to initialize GLOBAL_ncDsetList.\n");
            return -1;
        }
    }
    list_insert_after(GLOBAL_ncdsetList, (char *) &nc, sizeof(ncdset), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Get file info for a dataset and save in GLOBAL_ncdsetList for attribute handling
 */
int FORTRAN(ncf_add_dset)( int *ncid, int *setnum, char name[], char path[] )
{
    ncdset nc;

    /* code lifted liberally from ncdump.c Calls in nc library.*/

    char fillc;
    int i;    /* loop controls */
    int ia;
    int iv;
    int ilen;
    int nc_status;    /* return from netcdf calls */
    ncdim fdims;    /* name and size of dimension */
    ncatt att;    /* attribute */
    ncvar var;    /* variable */
    int bad_file_attr = 243;    /* matches merr_badfileatt in tmap_errors.parm*/
    int too_big_dim = 254;    /* matches merr_dimtoolarge in tmap_errors.parm*/
    int return_val=FERR_OK;
    size_t len;
    char **strarray;
    char *strptr;
    int idx;
	long big_size = 2147483647; /* max integer */

    ncf_init_dataset(&nc);
    strcpy(nc.fername, name);
    strcpy(nc.fullpath, path);
    nc.fer_dsetnum = *setnum;

    /*
     * get number of dimensions, number of variables, number of global
     * atts, and dimension id of unlimited dimension, if any
     */
    nc_status = nc_inq(*ncid, &nc.ndims, &nc.nvars, &nc.ngatts, &nc.recdim);
    if ( nc_status != NC_NOERR )
        return nc_status;

    /* get dimension info */
    if ( nc.ndims > 0 ) {
        for (i = 0; i < nc.ndims; i++) {
            nc_status = nc_inq_dim(*ncid, i, fdims.name, &fdims.size);
			if (fdims.size >= big_size)
			{  fprintf(stderr, "ERROR: Dimension %s is too large.\n", fdims.name);
			   return too_big_dim;
			}
            if ( nc_status != NC_NOERR )
                return nc_status;
            strcpy (nc.dims[i].name, fdims.name);
            nc.dims[i].size = fdims.size;
        }
    }

    /* Go ahead and create the variable list - dataset with nothing not very likely */
    nc.dsetvarlist = list_init(__FILE__, __LINE__);
    if ( nc.dsetvarlist == NULL ) {
        fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize variable list.\n");
        return -1;
    }

    /* get info on global attributes, treat as pseudo-variable . list of attributes */

    /* get global attributes */

    if ( nc.ngatts > 0 ) {
        ncf_init_variable(&var);
        strcpy(var.name, ".");
        var.type = NC_CHAR;
        var.outtype = NC_CHAR;
        var.varid = 0;
        var.natts = nc.ngatts;
        var.ndims = 1;
        var.dims[0] = 1;
        /* Create the list of attributes since there are going to be attributes added */
        var.varattlist = list_init(__FILE__, __LINE__);
        if ( var.varattlist == NULL ) {
            fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize GLOBAL attributes list.\n");
            return -1;
        }

        for (i = 0; i < nc.ngatts; i++) {
            ncf_init_attribute(&att);
            nc_status = nc_inq_attname(*ncid, NC_GLOBAL, i, att.name);
            if ( nc_status == NC_NOERR ) {
                att.attid = i+1;
                nc_status = nc_inq_att(*ncid, NC_GLOBAL, att.name, &att.type, &len);
                att.len = (int)len;
                if ( nc_status == NC_NOERR ) {
                    /* Set output flag. By default only the global history attribute is written.
                     * For string attributes, allocate one more than the att.len,
                     * for the null terminator for the string */
                    if ( strcmp(att.name,"history") == 0 ) {
                        att.outflag = 1;
                    }
                    if (att.len == 0) {    /* show 0-length attributes as empty strings */
                        att.type = NC_CHAR;
                        att.outtype = NC_CHAR;
                        att.len = 1;
                        att.string = (char *) FerMem_Malloc(2* sizeof(char), __FILE__, __LINE__);
                        strcpy(att.string, " ");
                    }
                    else {
                        switch (att.type) {
                        case NC_CHAR:
                            att.string = (char *) FerMem_Malloc((att.len+1)* sizeof(char), __FILE__, __LINE__);
                            nc_status = nc_get_att_text(*ncid, NC_GLOBAL, att.name, att.string);
                            if ( nc_status != NC_NOERR )
                                return nc_status;
                            break;
                        case NC_STRING:
                            /* len is the number of strings in this attribute */
                            strarray = (char **) FerMem_Malloc(len * sizeof(char *), __FILE__, __LINE__);
                            /* NetCDF allocates memory for each string in the returned array of strings */
                            nc_status = nc_get_att_string(*ncid, NC_GLOBAL, att.name, strarray);
                            if ( nc_status != NC_NOERR ) {
                                FerMem_Free(strarray, __FILE__, __LINE__);
                                att.string = NULL;
                                att.len = 0;
                                return nc_status;
                            }
                            /* allocate memory for a concatenation of all the strings */
                            att.len = 0;
                            for (idx = 0; idx < len; idx++)
                               att.len += strlen(strarray[idx]) + 1;
                            att.string = (char *) FerMem_Malloc(att.len * sizeof(char), __FILE__, __LINE__);
                            strptr = att.string;
                            /* copy the first string; keeping record of the end of the string */
                            strcpy(strptr, strarray[0]);
                            strptr += strlen(strarray[0]);
                            for (idx = 1; idx < len; idx++) {
                               /* append a new-line (takes the place of the previous null character) and this string */
                               strcpy(strptr, "\n");
                               strptr++;
                               strcpy(strptr, strarray[idx]);
                               strptr += strlen(strarray[idx]);
                            }
                            /* free memory for the strings allocated by NetCDF */
                            nc_free_string(len, strarray);
                            /* free memory for the string array */
                            FerMem_Free(strarray, __FILE__, __LINE__);
                            /* Say this is an NC_CHAR (instead of NC_STRING) attribute for Ferret internals */
                            /* Inside Ferret, character array and string attributes are the same */
                            att.type = NC_CHAR;
                            att.outtype = NC_CHAR;
                            break;
                        default:
                            att.vals = (double *) FerMem_Malloc(att.len * sizeof(double), __FILE__, __LINE__);
                            nc_status = nc_get_att_double(*ncid, NC_GLOBAL, att.name, att.vals);
                            if ( nc_status != NC_NOERR )
                                return nc_status;
                            break;
                        }
                    }
                }    /* end of the    if (nc_status == NC_NOERR) */
            }
            /*Save attribute in linked list of attributes for variable . (global attributes)*/
            list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);
        }        /* global attributes list complete */

        /*Save variable in linked list of variables for this dataset */
        list_insert_after(nc.dsetvarlist, (char *) &var, sizeof(ncvar), __FILE__, __LINE__);
    }

    /* get info on variables */
    for (iv = 0; iv < nc.nvars; iv++) {
        ncf_init_variable(&var);
        nc_status = nc_inq_var(*ncid, iv, var.name, &var.type, &var.ndims, var.dims, &var.natts);
        if ( nc_status != NC_NOERR )
            return nc_status;

        var.varid = iv+1;
        var.outtype = var.type;

        /* Is this a coordinate variable? If not a string, set the flag.
         * A multi-dimensional variable that shares a dimension name is not a coord. var that
         * is handled as a 1-D axis
         * A string-typed fvariable is not a coordinate var.
         */
        if ( nc.ndims > 0 ) {
            var.is_axis = FALSE;
            var.axis_dir = 0;
            i = 0;
            while ( (i < nc.ndims) && (var.is_axis == FALSE) ) {
                if ( strcasecmp(var.name, nc.dims[i].name) == 0 ) {
                    var.is_axis = TRUE;
                    if ( (var.type == NC_CHAR) || (var.type == NC_STRING) ) {
                       var.is_axis = FALSE;
/*
					   fprintf(stderr, "           *** NOTE: Axis %s is of type char or string\n", var.name);
                       fprintf(stderr, "           *** NOTE: A dummy axis of subscripts will be used\n");
*/
					}
                    if ( var.ndims > 1 )
                       var.is_axis = FALSE;
                }
                i = i + 1;
            }
        }

        /* get _FillValue attribute */
        nc_status =    nc_inq_att(*ncid,iv,"_FillValue",&att.type, &len);
        att.len = (int)len;
        if ( (nc_status == NC_NOERR) &&
                 (att.type == var.type) &&
                 (att.len == 1) ) {
            var.has_fillval = TRUE;
            if ( var.type == NC_CHAR ) {
                nc_status = nc_get_att_text(*ncid, iv, "_FillValue", &fillc);
                if ( nc_status != NC_NOERR ) {
                    return_val = bad_file_attr;
                    fprintf(stderr, "*** NOTE: Variable %s, error with character _FillValue\n", var.name);
                }
                else {
                    var.fillval = (double) fillc;
                }
            }
            else if ( var.type == NC_STRING ) {
                /* Only one string to read and return */
                nc_status = nc_get_att_string(*ncid, iv, "_FillValue", &strptr);
                if ( nc_status != NC_NOERR ) {
                    return_val = bad_file_attr;
                    fprintf(stderr, "*** NOTE: Variable %s, error with string _FillValue\n", var.name);
                }
                else {
                    /* Can only handle one character */
                    if ( strlen(strptr) > 1 )
                       fprintf(stderr, "*** NOTE: Variable %s, _FillValue truncated to '%c'\n", var.name, strptr[0]);
                    var.fillval = (double) (strptr[0]);
                    nc_free_string(1, &strptr);
                }
            }
            else {
                nc_status = nc_get_att_double(*ncid, iv, "_FillValue", &var.fillval);
            }
        }
        else    { /* set to default NC value*/
            var.has_fillval = FALSE;
            switch (var.type) {
            case NC_BYTE:
                /* don't do default fill-values for bytes, too risky */
                /* var.has_fillval = 0;  - already assigned to FALSE above */
                /* but give it some value to it is not random garbage */
                var.fillval = NC_FILL_BYTE;
                break;
            case NC_CHAR:
                var.fillval = NC_FILL_CHAR;
                break;
            case NC_SHORT:
                var.fillval = NC_FILL_SHORT;
                break;
            case NC_INT:
                var.fillval = NC_FILL_INT;
                break;
            case NC_FLOAT:
                var.fillval = NC_FILL_FLOAT;
                break;
            case NC_DOUBLE:
                var.fillval = NC_FILL_DOUBLE;
                break;
            case NC_STRING:
                var.fillval = NC_FILL_CHAR; /* '\0'; NC_FILL_STRING is (char *)"" - pointer to a static empty string - which is NOT what we want */
                break;
            default:
                break;
            }
        }
        var.all_outflag = 1;

        /* Go ahead and create the list for variable attributes */
        var.varattlist = list_init(__FILE__, __LINE__);
        if ( var.varattlist == NULL ) {
            fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize variable attributes list.\n");
            return -1;
        }

        /* get all variable attributes
         * For string attributes, allocate one more than the att.len,
         * for the null terminator for the string. See Netcdf User's Guide, nc_get_att_text.
         */

        for (ia = 0; ia < var.natts; ia++) {
            ncf_init_attribute(&att);
            nc_status = nc_inq_attname(*ncid, iv, ia, att.name);
            if (nc_status == NC_NOERR) {
                att.attid = ia+1;

                nc_status = nc_inq_att(*ncid, iv, att.name, &att.type, &len);
                att.len = (int)len;
                if ( nc_status == NC_NOERR ) {
                    if ( att.len == 0 ) {    /* set 0-length attributes to empty strings */
                        att.type = NC_CHAR;
                        att.outtype = NC_CHAR;
                        att.len = 1;
                        att.string = (char *) FerMem_Malloc(2*sizeof(char), __FILE__, __LINE__);
                        strcpy(att.string," ");
                    }
                    else {
                        switch ( att.type ) {
                        case NC_CHAR:
                            att.outtype = NC_CHAR;
                            /* Plus one for end-of-string delimiter. */
                            att.string = (char *) FerMem_Malloc((att.len+1)*sizeof(char), __FILE__, __LINE__);
                            strcpy (att.string, " ");
                            nc_status = nc_get_att_text(*ncid, iv, att.name, att.string);
                            if ( nc_status != NC_NOERR ) {    /* on error set attr to empty string*/
                                att.len = 1;
                                /* att.string already allocated to at least 2 */
                                strcpy(att.string, " ");
                                return_val = bad_file_attr;
                                fprintf(stderr, "*** NOTE: Variable %s, error on attribute %s\n", var.name, att.name);
                            }
                            else {
                                /* Ensure end-of-string delimiter because Netcdf API doesn't store automatically; it's up to the file's author. */
                                att.string[att.len] = '\0';

                                /* Check the actual string length (one example file has attribute units="m"
                                 * but gets att.len = 128 from nc_inq_att above) because user probably used
                                 * some arbitrarily large string buffer and partially populated the string
                                 * leaving the remainder filled with '\0'.    */
                                ilen = strlen(att.string);
                                if ( ilen < att.len )
                                    att.len = ilen;
                            }
                            break;
                        case NC_STRING:
                            att.outtype = NC_STRING;
                            /* len is the number of strings to read */
                            strarray = FerMem_Malloc(len * sizeof(char *), __FILE__, __LINE__);
                            nc_status = nc_get_att_string(*ncid, iv, att.name, strarray);
                            if ( nc_status != NC_NOERR ) {    /* on error set attr to empty string*/
                                att.len = 1;
                                att.string = (char *) FerMem_Malloc(2*sizeof(char), __FILE__, __LINE__);
                                strcpy(att.string, " ");
                                return_val = bad_file_attr;
                                fprintf(stderr, "*** NOTE: Variable %s, error on attribute %s\n", var.name, att.name);
                            }
                            else {
                                att.len = 0;
                                for (idx = 0; idx < len; idx++)
                                   att.len += strlen(strarray[idx]) + 1;
                                att.string = (char *) FerMem_Malloc(att.len*sizeof(char), __FILE__, __LINE__);
                                strptr = att.string;
                                strcpy(strptr, strarray[0]);
                                strptr += strlen(strarray[0]);
                                for (idx = 1; idx < len; idx++) {
                                   /* append a new-line (takes the place of the previous null character) and this string */
                                   strcpy(strptr, "\n");
                                   strptr++;
                                   strcat(strptr, strarray[idx]);
                                   strptr += strlen(strarray[idx]);
                                }
                                nc_free_string(len, strarray);
                            }
                            FerMem_Free(strarray, __FILE__, __LINE__);
                            /* Say this is an NC_CHAR (instead of NC_STRING) attribute for Ferret internals */
                            /* Inside Ferret, character array and string attributes are the same */
                            att.type = NC_CHAR;
                            att.outtype = NC_CHAR;
                        default:
#ifdef double_p
                            att.outtype = NC_DOUBLE;
#else
                            att.outtype = NC_FLOAT;
#endif
                            att.vals = (double *) FerMem_Malloc(att.len * sizeof(double), __FILE__, __LINE__);
                            nc_status = nc_get_att_double(*ncid, iv, att.name, att.vals);
                            if ( nc_status != NC_NOERR ) {    /* on error set attr to empty string*/
                                FerMem_Free(att.vals, __FILE__, __LINE__);
                                att.vals = NULL;
                                att.type = NC_CHAR;
                                att.outtype = NC_CHAR;
                                att.len = 1;
                                att.string = (char *) FerMem_Malloc((att.len+1)* sizeof(char), __FILE__, __LINE__);
                                strcpy (att.string, " ");
                                return_val = bad_file_attr;
                                fprintf(stderr, "*** NOTE: Variable %s, error on attribute %s\n", var.name, att.name);
                            }
                            break;
                        }
                    }

                    /* Initialize output flag. Attributes written by default by Ferret
                     * will be set to outflag = 1.
                     */
                    att.outflag = initialize_output_flag(att.name, var.is_axis);

                } /* end of the if (nc_status == NC_NOERR)    */
            }

            /*Save attribute in linked list of attributes for this variable */
            list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);
        }        /* variable attributes from file complete */

        /* If this is a coordinate variable, add an attribute orig_file_axname which
         * contains the axis name, and is used to preserve the original name if
         * Ferret detects a duplicate axis name and changes the axis name.*/
        if ( var.is_axis ) {
            var.natts = var.natts + 1;
            ncf_init_attribute(&att);
            strcpy (att.name, "orig_file_axname");
            att.attid = ia+1;
            att.type = NC_CHAR;
            att.len = strlen(var.name);
            att.string = (char *) FerMem_Malloc((att.len+1)*sizeof(char), __FILE__, __LINE__);
            strcpy (att.string,var.name);

            /* Ensure end-of-string delimiter because Netcdf API doesn't store automatically; it's up to the file's author. */
            att.string[att.len] = '\0';

            /* Output flag always false for this attribute */
            att.outflag = -1;

            /*Save attribute in linked list of attributes for this variable */
            list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);
        }

        /*Save variable in linked list of variables for this dataset */
        list_insert_after(nc.dsetvarlist, (char *) &var, sizeof(ncvar), __FILE__, __LINE__);
    }        /* variables list complete */

    /* Add dataset to global nc dataset linked list*/
    if ( GLOBAL_ncdsetList == NULL ) {
        GLOBAL_ncdsetList = list_init(__FILE__, __LINE__);
        if ( GLOBAL_ncdsetList == NULL ) {
            fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize GLOBAL_ncDsetList.\n");
            return -1;
        }
    }
    list_insert_after(GLOBAL_ncdsetList, (char *) &nc, sizeof(ncdset), __FILE__, __LINE__);

    return return_val;
}

/* ----
 * Initialize new dataset to contain a non-netcdf dataset
 * save in GLOBAL_ncdsetList for attribute handling
 */
int FORTRAN(ncf_init_other_dset)( int *setnum, char name[], char path[] )
{
    ncdset nc;
    ncvar var; /* variable */
    ncatt att; /* attribute */

    ncf_init_dataset(&nc);
    strcpy(nc.fername, name);
    strcpy(nc.fullpath, path);
    nc.fer_dsetnum = *setnum;
    nc.ngatts = 1;

    /* set up pseudo-variable . the list of variables */
    ncf_init_variable(&var);
    strcpy(var.name, ".");
    var.type = NC_CHAR;
    var.outtype = NC_CHAR;
    var.varid = 0;
    var.natts = 1;
    var.ndims = 0;

    /* set one global attribute, history */
    ncf_init_attribute(&att);
    att.outflag = 1;
    att.type = NC_CHAR;
    att.outtype = NC_CHAR;
    att.outflag = 0;
    att.attid = 1;
    att.len = strlen(name);
    strcpy(att.name, "history");
    att.string = (char *) FerMem_Malloc((att.len+1)* sizeof(char), __FILE__, __LINE__);
    strcpy(att.string, name);

    /* Save attribute in linked list of attributes for variable */
    var.varattlist = list_init(__FILE__, __LINE__);
    if ( var.varattlist == NULL ) {
        fprintf(stderr, "ERROR: ncf_init_other_dset: Unable to initialize GLOBAL attributes list.\n");
        return -1;
    }
    list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    /* global attributes list complete */

    /*Save variable in linked list of variables for this dataset */
    nc.dsetvarlist = list_init(__FILE__, __LINE__);
    if ( nc.dsetvarlist == NULL ) {
        fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize variable list.\n");
        return -1;
    }
    list_insert_after(nc.dsetvarlist, (char *) &var, sizeof(ncvar), __FILE__, __LINE__);

    /* Add dataset to global nc dataset linked list */
    if ( GLOBAL_ncdsetList == NULL ) {
        GLOBAL_ncdsetList = list_init(__FILE__, __LINE__);
        if ( GLOBAL_ncdsetList == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize GLOBAL_ncDsetList.\n");
            return -1;
        }
    }
    list_insert_after(GLOBAL_ncdsetList, (char *) &nc, sizeof(ncdset), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Find a dataset based on an integer id and return the pointer to the dataset.
 */
static ncdset *ncf_get_ds_ptr( int *dset )
{
    int status;
    ncdset *nc_ptr;

    status = list_traverse(GLOBAL_ncdsetList, (char *) dset, NCF_ListTraverse_FoundDsetID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return NULL;

    nc_ptr = (ncdset *) list_curr(GLOBAL_ncdsetList);
    return nc_ptr;
}

/* ----
 * Find a dataset based on its integer ID and return a pointer to its variable list
 */
static LIST *ncf_get_ds_varlist( int *dset )
{
    ncdset *nc_ptr;
    LIST *varlist;

    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return NULL;

    varlist = nc_ptr->dsetvarlist;
    return varlist;
}

/* ----
 * Find a variable in a dataset based on its dataset integer ID and variable integer ID.
 * Return a pointer to the variable.
 */
static ncvar *ncf_get_ds_var_ptr( int *dset, int *varid )
{
    LIST *varlist;
    int status;
    ncvar *var_ptr;

    varlist = ncf_get_ds_varlist(dset);
    if ( varlist == NULL )
        return NULL;
    status = list_traverse(varlist, (char *) varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return NULL;

    var_ptr = (ncvar *) list_curr(varlist);
    return var_ptr;
}

/* ----
 * Initialize an ncatt to default values just to make sure
 * there are no random garbage values in the structure.
 */
static void ncf_init_attribute( ncatt *att_ptr )
{
    /* Just set everything to 0/NULL/FALSE/empty string */
    memset(att_ptr, 0, sizeof(ncatt));
    /* No fields to adjust */
}

/* ----
 * Free the fields in an ncatt, then the ncatt itself
 * For using with list_free.
 */
static void ncf_free_attribute( char *attptr ) {
    ncatt *att;

    att = (ncatt *)attptr;
    if ( att->string != NULL ) {
        FerMem_Free(att->string, __FILE__, __LINE__);
        att->string = NULL;
    }
    if ( att->vals != NULL ) {
        FerMem_Free(att->vals, __FILE__, __LINE__);
        att->vals = NULL;
    }
    /* paranoia */
    memset(att, 0, sizeof(ncatt));
    FerMem_Free(att, __FILE__, __LINE__);
}

/* ----
 * Deallocates ncatts for a ncvar.
 */
static void ncf_free_attlist( ncvar* varptr )
{
    if (varptr == NULL)
        return;
    if ( varptr->varattlist == NULL )
        return;

    /*
     * Free the attribute list using ncf_free_attribute
     * to free each data element in the list
     */
    list_free(varptr->varattlist, ncf_free_attribute, __FILE__, __LINE__);
    varptr->varattlist = NULL;
    varptr->natts = 0;
}

/* ----
 * Initialize an ncvar to default values just to make sure
 * there are no random garbage values in the structure.
 */
static void ncf_init_variable( ncvar *var_ptr )
{
    /* Just set everything to 0/NULL/FALSE/empty string */
    memset(var_ptr, 0, sizeof(ncvar));
    /* Adjust a few fields */
    var_ptr->ndims = 6;
    var_ptr->all_outflag = 1;
#ifdef double_p
    var_ptr->fillval = NC_FILL_DOUBLE;
#else
    var_ptr->fillval = NC_FILL_FLOAT;
#endif
}

/* ----
 * Free the fields in an ncvar, then the ncvar itself
 * For using with list_free.
 * This does NOT reset the varid's or reset the number
 * of vars in the dataset.
 */
static void ncf_free_variable( char *varptr )
{
    ncvar *var;

    if (varptr == NULL)
        return;
    var = (ncvar *)varptr;
    /* Free the list of attributes */
    ncf_free_attlist(var);
    /* Free the list of ncagg_var_descr */
    if ( var->varagglist != NULL ) {
        list_free(var->varagglist, LIST_DEALLOC, __FILE__, __LINE__);
        var->varagglist = NULL;
    }
    /* Free the list of uvarGrids */
    if ( var->uvarGridList != NULL ) {
        list_free(var->uvarGridList, LIST_DEALLOC, __FILE__, __LINE__);
        var->uvarGridList = NULL;
    }
    /* paranoia */
    memset(var, 0, sizeof(ncvar));
    /* Free the variable itself */
    FerMem_Free(var, __FILE__, __LINE__);
}

/* ----
 * Initialize an ncdset to default values just to make sure
 * there are no random garbage values in the structure.
 */
static void ncf_init_dataset( ncdset *dset_ptr )
{
    /* Just set everything to 0/NULL/FALSE/empty string */
    memset(dset_ptr, 0, sizeof(ncdset));
    /* Adjust a few fields */
    dset_ptr->ndims = 6;
    dset_ptr->recdim = -1;
}

/* ----
 * Free the fields in an ncdset, then the ncdset itself.
 * For use with list_free.
 * Does not remove the ncdset from the global list.
 */
static void ncf_free_dataset( char *nc_ptr )
{
    ncdset *nc;

    if ( nc_ptr == NULL )
        return;
    nc = (ncdset *) nc_ptr;

    /* Free the ncvar's associated with this dataset */
    if ( nc->dsetvarlist != NULL ) {
       list_free(nc->dsetvarlist, ncf_free_variable, __FILE__, __LINE__);
       nc->dsetvarlist = NULL;
    }

    /* Free any ncagg structs about aggregated dataset members */
    if ( nc->agg_dsetlist != NULL ) {
       list_free(nc->agg_dsetlist, LIST_DEALLOC, __FILE__, __LINE__);
       nc->agg_dsetlist = NULL;
    }

    /* paranoia */
    memset(nc, 0, sizeof(ncdset));
    /* Free the ncdset itself */
    FerMem_Free(nc, __FILE__, __LINE__);
}

/* ----
 * Remove all datasets for the global datasets list
 * and remove the global datasets list.
 */
void FORTRAN(ncf_datasets_list_clear)( void )
{
    if ( GLOBAL_ncdsetList != NULL ) {
        list_free(GLOBAL_ncdsetList, ncf_free_dataset, __FILE__, __LINE__);
        GLOBAL_ncdsetList = NULL;
    }
}

/* ----
 * Remove a dataset from the global dataset list
 */
int FORTRAN(ncf_delete_dset)( int *dset )
{
    ncdset *nc_ptr;

    /* Find the dataset */
    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    /* Free the list of variables for this dataset */
    if ( nc_ptr->dsetvarlist != NULL ) {
        list_free(nc_ptr->dsetvarlist, ncf_free_variable, __FILE__, __LINE__);
        nc_ptr->dsetvarlist = NULL;
    }
    nc_ptr->nvars = 0;
    nc_ptr->ngatts = 0;

    /* Free any ncagg structs about aggregated dataset members */
    if ( nc_ptr->agg_dsetlist != NULL ) {
       list_free(nc_ptr->agg_dsetlist, LIST_DEALLOC, __FILE__, __LINE__);
       nc_ptr->agg_dsetlist = NULL;
    }
    nc_ptr->num_agg_members = 0;

    /* Remove the dataset from dataset list */
    if ( nc_ptr != (ncdset *) list_remove_curr(GLOBAL_ncdsetList, __FILE__, __LINE__) ) {
        fprintf(stderr, "ERROR: ncf_delete_dset: Unexpected mismatch of current dataset in global list.\n");
        return -1;
    }

    /* paranoia */
    memset(nc_ptr, 0, sizeof(ncdset));
    /* Free the ncdset itself */
    FerMem_Free(nc_ptr, __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Add a new variable to a dataset.
 * If varid is < 0, set it to nvars+1 for this dataset, return varid
 * and store -1*varid as the user variable ID (uvarid)
 */
int FORTRAN(ncf_add_var)( int *dset, int *varid, int *type, int *coordvar,
                          char *varname, char title[], char units[], double *bad )
{
    ncdset *nc_ptr;
    ncatt att;
    ncvar var;
    ncagg_var_descr vdescr;
    int status;
    int newvar;
    LIST *vlist;

    /*
     * Get the dataset pointer.
     */
    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of variables.    See if this variable already exists.
     */
    newvar = FALSE;
    vlist = nc_ptr->dsetvarlist;
    status = list_traverse(vlist, varname, NCF_ListTraverse_FoundVarName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK ) {
        newvar = TRUE;
    }

    if ( ! newvar ) {
        /*
         * If this variable is not new, remove the old definition of it.
         * This decrements the appropriate varids as well as
         * the number of vars in the dataset.
         */
        FORTRAN(ncf_delete_var)(dset, varname);
    }

    nc_ptr->nvars = nc_ptr->nvars + 1;

    /*
     * Set variable structure and insert the new variable
     * at the end of the variable list.
     */
    ncf_init_variable(&var);
    strcpy(var.name,varname);
    var.type = *type;
    var.outtype = *type;
    if ( *varid < 0 ) {
        /* user variable (aka "LET") */
        var.uvarid = -1 * (*varid);    /* value of uvar as found in Ferret */
        if ( *dset == PDSET_UVARS ) {
            /* for global uvars,    varid always matches uvarid */
            /* which means that gaps may occur in the varid sequence */
             var.varid = var.uvarid;
        }
        else {
            /* for LET/D uvars, varid is the var count */
            /* ==> gaps in varid must be compacted when LET/D vars are deleted */
            var.varid = nc_ptr->nvars;
        }
    }
    else {
        /* file variable */
        var.varid = nc_ptr->nvars;
        var.uvarid = 0;     /* 0 signals a file var */
    }

    var.is_axis = *coordvar;

    /* Set up initial set of attributes*/
    var.varattlist = list_init(__FILE__, __LINE__);
    if ( var.varattlist == NULL ) {
        fprintf(stderr, "ERROR: ncf_add_var: Unable to initialize attributes list.\n");
        return -1;
    }

    /* Save the long_name, all variables.
     * For string attributes, allocate one more than the att.len,
     * for the null terminator for the string
     */
    var.natts = var.natts+1;
    ncf_init_attribute(&att);
    strcpy(att.name, "long_name");
    att.type = NC_CHAR;
    att.outtype = NC_CHAR;
    att.attid = var.natts;
    att.outflag = 1;
    att.len = strlen(title);
    att.string = (char *) FerMem_Malloc((att.len+1)* sizeof(char), __FILE__, __LINE__);
    strcpy(att.string, title);
    /* Save attribute in linked list of attributes for this variable */
    list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    /* Now the units, if given
     * For the units string, allocate one more than the att.len,
     * for the null terminator for the string
     */
    if (strlen(units) > 0 ) {
        var.natts = var.natts+1;
        ncf_init_attribute(&att);
        att.attid = var.natts;
        strcpy(att.name, "units");
        att.len = strlen(units);
        att.outflag = 1;
        att.type = NC_CHAR;
        att.outtype = NC_CHAR;
        att.string = (char *) FerMem_Malloc((att.len+1)* sizeof(char), __FILE__, __LINE__);
        strcpy(att.string, units);
        /* Save attribute in linked list of attributes for this variable */
        list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);
    }

    /* Now the missing_value, for numeric variables */
    var.natts = var.natts+1;
    var.fillval = *bad;
    ncf_init_attribute(&att);
    att.attid = var.natts;
    strcpy(att.name,"missing_value");
    att.len = 1;
#ifdef double_p
    att.type = NC_DOUBLE;
    att.outtype = NC_DOUBLE;
#else
    att.type = NC_FLOAT;
    att.outtype = NC_FLOAT;
#endif
    att.vals = (double *) FerMem_Malloc(att.len * sizeof(double), __FILE__, __LINE__);
    att.vals[0] = *bad;
    /* Initialize output flag. Attributes written by default by Ferret
     * will be set to outflag = 1.
     */
    att.outflag = initialize_output_flag (att.name, var.is_axis);
    /*Save attribute in linked list of attributes for this variable */
    list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    /* If this is an aggregate dataset, initialize the list of member-info
     * for the variable. The values will be filled in later.
     */
    var.varagglist = list_init(__FILE__, __LINE__);
    if ( var.varagglist == NULL ) {
        fprintf(stderr, "ERROR: ncf_add_var: Unable to initialize aggregate info list.\n");
        return -1;
    }

    vdescr.imemb = 0;
    vdescr.gnum = 0;
    list_insert_after(var.varagglist, (char *) &vdescr, sizeof(ncagg_var_descr), __FILE__, __LINE__);

    /* if it's a uvar, then initialize a grid LIST for it */
    if ( var.uvarid != 0 ) {
        var.uvarGridList = list_init(__FILE__, __LINE__);
        if ( var.uvarGridList == NULL ) {
            fprintf(stderr, "ERROR: ncf_add_var: Unable to initialize uvar grid list.\n");
            return -1;
        }
    }

    /* Save variable in linked list of variables for this dataset */
    list_mvrear(nc_ptr->dsetvarlist);
    list_insert_after(nc_ptr->dsetvarlist, (char *) &var, sizeof(ncvar), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Add a new variable to the pseudo user-defined coordinate variable dataset.
 */
int FORTRAN(ncf_add_coord_var)( int *dset, int *varid, int *type, int *coordvar,
                                char varname[], char units[], double *bad )
{
    ncdset *nc_ptr;
    ncatt att;
    ncvar var;
    int status;
    int newvar;
    LIST *vlist;
    ncvar *var_ptr;

    /*
     * Get the dataset pointer.
     */
    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of variables.    See if this variable already exists.
     */
    newvar = FALSE;
    vlist = nc_ptr->dsetvarlist;
    status = list_traverse(vlist, varname, NCF_ListTraverse_FoundVarName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK ) {
        newvar = TRUE;
    }

    if ( ! newvar ) {
        /*
         * If this variable is not new, remove the old definition of it.
         * Do not decrement varids or nvars !
         */
        var_ptr = (ncvar *) list_remove_curr(vlist, __FILE__, __LINE__);
        ncf_free_variable((char *) var_ptr);
    }

    nc_ptr->nvars = nc_ptr->nvars + 1;

    /*
     * Set variable structure and insert the new variable at the end of the
     * variable list.
     */
    ncf_init_variable(&var);
    strcpy(var.name, varname);
    var.type = *type;
    var.outtype = *type;
    var.varid = nc_ptr->nvars;
    *varid = nc_ptr->nvars;
    var.is_axis = *coordvar;
    var.fillval = *bad;
    var.varattlist = list_init(__FILE__, __LINE__);
    if ( var.varattlist == NULL ) {
        fprintf(stderr, "ERROR: ncf_add_coord_var: Unable to initialize attributes list.\n");
        return -1;
    }

    /* Set up initial set of attributes*/

    /* Units, if given
     * For the units string, allocate one more than the att.len,
     * for the null terminator for the string
     */
    if ( strlen(units) > 0 ) {
        var.natts = var.natts+1;
        ncf_init_attribute(&att);
        att.attid = var.natts;
        strcpy(att.name, "units");
        att.len = strlen(units);
        att.outflag = 1;
        att.type = NC_CHAR;
        att.outtype = NC_CHAR;
        att.string = (char *) FerMem_Malloc((att.len+1)* sizeof(char), __FILE__, __LINE__);
        strcpy(att.string, units);
        list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);
    }

    /* Save variable in linked list of variables for this dataset */
    list_mvrear(nc_ptr->dsetvarlist);
    list_insert_after(nc_ptr->dsetvarlist, (char *) &var, sizeof(ncvar), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Find a variable based on its variable ID and dataset ID
 * Add a new numeric attribute.
 */
int FORTRAN(ncf_add_var_num_att)( int *dset, int *varid, char attname[],
                                  int *attype, int *attlen, int *outflag, DFTYPE *vals )
{
    ncvar *var_ptr;
    LIST *varattlist;
    ncatt *att_ptr;
    ncatt att;
    int status;
    int i;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of attributes for the variable in the dataset
     * If the attribute is already defined, return -1* attid
     */
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status == LIST_OK ) {
        att_ptr = (ncatt *) list_curr(varattlist);
        return (-1 * att_ptr->attid);
    }

    /* Increment number of attributes. */

    var_ptr->natts = var_ptr->natts + 1;

    /*
     * Set attribute structure and insert the new attribute at
     * the end of the attribute list.
     */
    ncf_init_attribute(&att);
    strcpy(att.name, attname);
    att.attid = var_ptr->natts;
    att.type = *attype;
#ifdef double_p
    att.outtype = NC_FLOAT;
#else
    att.outtype = NC_DOUBLE;
#endif
    att.len = *attlen;
    att.outflag = *outflag;
    att.vals = (double *) FerMem_Malloc(*attlen * sizeof(double), __FILE__, __LINE__);

    for (i = 0; i < *attlen; i++) {
        att.vals[i] = vals[i];
    }

    /* Save attribute in linked list of attributes for this variable */
    list_insert_after(varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Find a variable based on its variable ID and dataset ID
 * Add a new numeric attribute.
 */
int FORTRAN(ncf_add_var_num_att_dp)( int *dset, int *varid, char attname[],
                                     int *attype, int *attlen, int *outflag, double *vals )
{
    ncvar *var_ptr;
    ncatt *att_ptr;
    ncatt att;
    int status;
    int i;
    LIST *varattlist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of attributes for the variable in the dataset
     * If the attribute is already defined, return -1* attid
     */
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status == LIST_OK ) {
        att_ptr = (ncatt *) list_curr(varattlist);

        return (-1 * att_ptr->attid);
    }

    /* Increment number of attributes. */
    var_ptr->natts = var_ptr->natts + 1;

    /*
     * Set attribute structure and insert the new attribute at
     * the end of the attribute list.
     */
    ncf_init_attribute(&att);
    strcpy(att.name, attname);
    att.attid = var_ptr->natts;
    att.type = *attype;
    att.outtype = NC_DOUBLE;
    att.len = *attlen;
    att.outflag = *outflag;
    att.vals = (double *) FerMem_Malloc(*attlen * sizeof(double), __FILE__, __LINE__);
    for (i = 0; i < *attlen; i++) {
        att.vals[i] = vals[i];
    }

    /*Save attribute in linked list of attributes for this variable */
    list_insert_after(varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Find a variable    based on its variable ID and dataset ID
 * Add a new string attribute.
 */
int FORTRAN(ncf_add_var_str_att)( int *dset, int *varid, char attname[], int *attype,
                                  int *attlen, int *outflag, char attstring[])
{
    ncdset *nc_ptr;
    ncvar *var_ptr;
    ncatt *att_ptr;
    ncatt att;
    int status;
    LIST *varlist;
    LIST *varattlist;

    /*
     * Get the dataset pointer.
     */
    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    varlist = nc_ptr->dsetvarlist;
    status = list_traverse(varlist, (char *) varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    var_ptr = (ncvar *) list_curr(varlist);

    /* Save attribute in linked list of attributes for variable */
    if ( var_ptr->varattlist == NULL ) {
        var_ptr->varattlist = list_init(__FILE__, __LINE__);
        if ( var_ptr->varattlist == NULL ) {
            fprintf(stderr, "ERROR: add_var_str_att: Unable to initialize attributes list.\n");
            return -1;
        }
    }
    else {
        varattlist = var_ptr->varattlist;
        /*
         * Get the list of attributes for the variable in the dataset
         * If the attribute is already defined, return -1* attid
         */
        status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
        if ( status == LIST_OK ) {
            att_ptr = (ncatt *) list_curr(varattlist);
            return (-1 * att_ptr->attid);
        }
    }

    /* Increment number of attributes. */
    if ( *varid == 0 ) {
        nc_ptr->ngatts = nc_ptr->ngatts + 1;
    }
    var_ptr->natts = var_ptr->natts + 1;

    /*
     * Set attribute structure and insert the new attribute at
     * the end of the attribute list.
     *
     *    For string attributes, allocate one more than the att.len,
     *    for the null terminator for the string
     */
    ncf_init_attribute(&att);
    strcpy(att.name,attname);
    att.attid = var_ptr->natts;
    att.type = *attype;
    att.outtype = NC_CHAR;
    att.len = *attlen;
    att.outflag = *outflag;
    att.string = (char *) FerMem_Malloc((att.len+1)* sizeof(char), __FILE__, __LINE__);
    strcpy(att.string, attstring);

    /*Save attribute in linked list of attributes for this variable */
    list_insert_after(var_ptr->varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Find a variable based on its variable ID and dataset ID
 * Replace the variable name with the new one passed in.
 */
int FORTRAN(ncf_rename_var)( int *dset, int *varid, char newvarname[] )
{
    ncvar *var_ptr;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /* Insert the new name. */
    strcpy(var_ptr->name, newvarname);

    return FERR_OK;
}

/* ----
 * Find a dimension in the datset using dataset ID
 * Replace the dimension name with the new one passed in.
 */
int FORTRAN(ncf_rename_dim)( int *dset, int *dimid, char newdimname[] )
{
    ncdset *nc_ptr;

     /*
     * Get the dataset pointer.
     */
    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    /* Insert the new name. */
    strcpy(nc_ptr->dims[*dimid-1].name, newdimname);

    return FERR_OK;
}

/* ----
 * Find an attribute based on its variable ID and dataset ID
 * Replace the type, length, and/or value(s).
 */
int FORTRAN(ncf_repl_var_att)( int *dset, int *varid, char attname[], int *attype,
                               int *attlen, DFTYPE *vals, char attstring[] )
{
    ncatt *att_ptr;
    ncvar *var_ptr;
    int status;
    int i;
    LIST *varattlist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of attributes for the variable in the dataset
     * If the attribute is not defined, return
     */
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    att_ptr = (ncatt *) list_curr(varattlist);

    /*
     * Free the memory used by the string or values
     */
    if ( att_ptr->string != NULL ) {
        FerMem_Free(att_ptr->string, __FILE__, __LINE__);
        att_ptr->string = NULL;
    }
    if ( att_ptr->vals != NULL ) {
        FerMem_Free(att_ptr->vals, __FILE__, __LINE__);
        att_ptr->vals = NULL;
    }

    /*
     * Keep the name and ID. Reset type, length, and values
     * For string attributes, allocate one more than the att.len
     * for the null terminator for the string
     */
    att_ptr->type = *attype;
    att_ptr->outtype = NC_FLOAT;
    att_ptr->len = *attlen;
    if (*attlen == 0) { /* set 0-length attributes to empty strings */
        att_ptr->type = NC_CHAR;
        att_ptr->outtype = NC_CHAR;
        att_ptr->len = 1;
        att_ptr->string = (char *) FerMem_Malloc(2*sizeof(char), __FILE__, __LINE__);
        strcpy(att_ptr->string," ");
    }
    else {
        switch (*attype) {
        case NC_CHAR:
        case NC_STRING:
            att_ptr->string = (char *) FerMem_Malloc((*attlen+1)* sizeof(char), __FILE__, __LINE__);
            strcpy(att_ptr->string,attstring);
            break;
        default:
            att_ptr->vals = (double *) FerMem_Malloc(*attlen * sizeof(double), __FILE__, __LINE__);
            for (i = 0; i < *attlen; i++) {
                att_ptr->vals[i] = vals[i];
            }
            break;
        }
    }

    return FERR_OK;
}

/* ----
 * Find an attribute based on its variable ID and dataset ID
 * Replace the type, length, and/or value(s).
 */
int FORTRAN(ncf_repl_var_att_dp)( int *dset, int *varid, char attname[], int *attype,
                                  int *attlen, double *vals, char attstring[] )
{
    ncatt *att_ptr;
    ncvar *var_ptr;
    int status;
    int i;
    LIST *varattlist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of attributes for the variable in the dataset
     * If the attribute is not defined, return
     */
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    /*
     * Get the attribute.
     */
    att_ptr = (ncatt *) list_curr(varattlist);

    /*
     * Free the memory used by the string or values
     */
    if ( att_ptr->string != NULL ) {
        FerMem_Free(att_ptr->string, __FILE__, __LINE__);
        att_ptr->string = NULL;
    }
    if ( att_ptr->vals != NULL ) {
        FerMem_Free(att_ptr->vals, __FILE__, __LINE__);
        att_ptr->vals = NULL;
    }

    /*
     * Keep the name and ID. Reset type, length, and values
     *    For string attributes, allocate one more than the att.len,
     *    presumably for the null terminator for the string (?)
     */

    att_ptr->type = *attype;
#ifdef double_p
    att_ptr->outtype = NC_DOUBLE;
#else
    att_ptr->outtype = NC_FLOAT;
#endif
    att_ptr->len = *attlen;

    if ( *attlen == 0 ) {    /* set 0-length attributes to empty strings */
        att_ptr->type = NC_CHAR;
        att_ptr->outtype = NC_CHAR;
        att_ptr->len = 1;
        att_ptr->string = (char *) FerMem_Malloc(2* sizeof(char), __FILE__, __LINE__);
        strcpy(att_ptr->string," ");
    }
    else {
        switch (*attype) {
        case NC_CHAR:
        case NC_STRING:
            att_ptr->string = (char *) FerMem_Malloc((*attlen+1)* sizeof(char), __FILE__, __LINE__);
            strcpy(att_ptr->string,attstring);
            break;
        default:
            att_ptr->vals = (double *) FerMem_Malloc(*attlen * sizeof(double), __FILE__, __LINE__);
            for (i = 0; i < *attlen; i++) {
                att_ptr->vals[i] = vals[i];
            }
            break;
        }
    }

    return FERR_OK;
}

/* ----
 * Find an attribute based on its variable ID and dataset ID
 * Delete it.
 */
int FORTRAN(ncf_delete_var_att)( int *dset, int *varid, char attname[] )
{
    ncatt *att_ptr;
    ncvar *var_ptr;
    int status;
    int att_to_remove;
    LIST *varattlist;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of attributes for the variable in the dataset
     * If the attribute is not defined, return
     */
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the attribute.
     */
    status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    /* Remove the attribute, saving its attribute number */
    att_ptr = (ncatt *) list_remove_curr(varattlist, __FILE__, __LINE__);
    att_to_remove = att_ptr->attid;
    ncf_free_attribute((char *) att_ptr);

    /* Decrement number of attributes for the variable. */
    var_ptr->natts = var_ptr->natts - 1;

    /*
     * Reset the attribute id for remaining attributes
     */
    list_mvfront(varattlist);
    do {
        att_ptr = (ncatt *) list_curr(varattlist);
        if ( (att_ptr != NULL) && (att_ptr->attid > att_to_remove) ) {
            att_ptr->attid = att_ptr->attid -1;
        }
    } while ( list_mvnext(varattlist) != NULL );

    return FERR_OK;
}

/* ----
 * Find an attribute based on its variable ID and dataset ID
 * Change its output flag: 1=output it, 0=dont.
 */
int FORTRAN(ncf_set_att_flag)( int *dset, int *varid, char attname[], int *attoutflag )
{
    ncatt *att_ptr;
    ncvar *var_ptr;
    int status;
    LIST *varattlist;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of attributes for the variable in the dataset
     * If the attribute is not defined, return
     */
    if ( var_ptr->natts < 1 )
        return ATOM_NOT_FOUND;
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL )
        return ATOM_NOT_FOUND;

    /* Get the attribute */
    status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    att_ptr = (ncatt *) list_curr(varattlist);

    /*
     * Keep the attribute as is, but reset its output flag.
     */
    att_ptr->outflag = *attoutflag;

    return FERR_OK;
}

/* ----
 * Find variable based on its variable ID and dataset ID
 * Change the variable flag:
 * 1=output no attributes,
 * 0=check individual attribute output flags,
 * 2=write all attributes, except any internal Ferret
 *     attributes, marked with outflag=-1.
 * 3=reset attr flags to defaults
 */
int FORTRAN(ncf_set_var_out_flag)( int *dset, int *varid, int *all_outflag )
{
    ncvar *var_ptr;
    ncatt *att_ptr;
    LIST *varattlist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Keep the default if there are no attributes
     */
    if ( var_ptr->natts < 1 ) {
        var_ptr->all_outflag = 1;
        return FERR_OK;
    }
    varattlist = var_ptr->varattlist;
    if ( varattlist == NULL ) {
        var_ptr->all_outflag = 1;
        return FERR_OK;
    }

    /*
     * Reset the variable output flag.
     */
    var_ptr->all_outflag = *all_outflag;
    if ( *all_outflag == 0 ) {

        /*
         * reset the output flag for each attribute
         */
        list_mvfront(varattlist);
        do {
            att_ptr = (ncatt *) list_curr(varattlist);
            if ( att_ptr != NULL ) {
                att_ptr->outflag = 0;
            }
        } while ( list_mvnext(varattlist) != NULL );

    }
    else if ( *all_outflag == 2 ) {

        /*
         * reset the output flag for each attribute
         */
        list_mvfront(varattlist);
        do {
            att_ptr = (ncatt *) list_curr(varattlist);
            if ( (att_ptr != NULL) && (att_ptr->outflag != -1) ) {
                att_ptr->outflag = 1;
            }
        } while ( list_mvnext(varattlist) != NULL );

    }
    else if ( *all_outflag == 3 ) {

        /*
         * reset the output flag for each attribute to the default Ferret value
         */
        list_mvfront(varattlist);
        do {
            att_ptr = (ncatt *) list_curr(varattlist);
            if ( att_ptr != NULL ) {
                /*
                 * Reset the attribute output flag to the Ferret default value
                 * (output missing flag, etc, but not nonstd attributes from
                 * the input file or user definitions.)
                 */
                att_ptr->outflag = initialize_output_flag(att_ptr->name, var_ptr->is_axis);
            }
        } while ( list_mvnext(varattlist) != NULL );

    }

    return FERR_OK;
}

/* ----
 * Find variable based on its variable ID and dataset ID
 * Change the variable output type.
 */
int FORTRAN(ncf_set_var_outtype)( int *dset, int *varid, int *outtype )
{
    ncvar *var_ptr;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /* Reset the variable output type */
    var_ptr->outtype = *outtype;

    return FERR_OK;
}


/* ----
 * Find variable based on its variable ID and dataset ID
 * Check that its a coordinate variable and set the axis direction.
 */
int FORTRAN(ncf_set_axdir)( int *dset, int *varid, int *axdir )
{
    ncvar *var_ptr;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;
    if ( ! var_ptr->is_axis )
        return ATOM_NOT_FOUND;

    /* Reset the variable output type */
    var_ptr->axis_dir = *axdir;

    return FERR_OK;
}

/* ----
 * Find an attribute based on its dataset ID, variable ID and attribute ID
 * Add the attribute to variable 2 in dataset 2
 */
int FORTRAN(ncf_transfer_att)( int *dset1, int *varid1, int *iatt, int *dset2, int *varid2 )
{
    ncatt *att_ptr1;
    ncatt att;
    ncvar *var_ptr1;
    ncvar *var_ptr2;
    int status;
    int i;
    LIST *varattlist1;

    /* Get the variable varid1 in dset1 */
    var_ptr1 = ncf_get_ds_var_ptr(dset1, varid1);
    if ( var_ptr1 == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of attributes for the variable varid1
     * If the attribute is not defined, return
     */
    varattlist1 = var_ptr1->varattlist;
    if ( varattlist1 == NULL )
        return ATOM_NOT_FOUND;
    status = list_traverse(varattlist1, (char *) iatt, NCF_ListTraverse_FoundVarAttID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    /* Get the attribute */
    att_ptr1 = (ncatt *) list_curr(varattlist1);

    /* Get the variable varid2 in dset2 */
    var_ptr2 = ncf_get_ds_var_ptr(dset2, varid2);
    if ( var_ptr2 == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of attributes for the variable varid2
     */
    if ( var_ptr2->varattlist == NULL ) {
        var_ptr2->varattlist = list_init(__FILE__, __LINE__);
        if ( var_ptr2->varattlist == NULL ) {
            fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize variable attributes list.\n");
            return -1;
        }
    }

    /* Increment number of attributes for varid2 */
    var_ptr2->natts = var_ptr2->natts + 1;

    /*
     * Set attribute structure and insert the new attribute at
     * the end of the attribute list.
     * For string attributes, allocate one more than the att.len,
     * for the null terminator for the string
     */
    ncf_init_attribute(&att);
    strcpy(att.name, att_ptr1->name);
    att.attid = var_ptr2->natts;
    att.type = att_ptr1->type;
    att.outtype = att_ptr1->type;
    att.len = att_ptr1->len;
    att.outflag = att_ptr1->outflag;

    if ( (att_ptr1->type == NC_CHAR) || (att_ptr1->type == NC_STRING) ) {
        att.string = (char *) FerMem_Malloc((att_ptr1->len+1)* sizeof(char), __FILE__, __LINE__);
        strcpy(att.string, att_ptr1->string);
    }
    else {
        att.vals = (double *) FerMem_Malloc(att_ptr1->len * sizeof(double), __FILE__, __LINE__);
        for (i = 0; i<att_ptr1->len;i++ ) {
            att.vals[i] = att_ptr1->vals[i];
        }
    }

    /* Save attribute in linked list of attributes for this variable */
    list_insert_after(var_ptr2->varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Find variable based on the dataset ID and variable name
 * Delete it.    This includes resetting the varids.
 */
int FORTRAN(ncf_delete_var)( int *dset, char *varname )
{
    ncdset *nc_ptr;
    ncvar *var_ptr;
    int status;
    int ivar;
    LIST *varlist;

    /* Find the dataset based on its integer ID */
    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of variables. Find varname in the dataset.
     */
    varlist = nc_ptr->dsetvarlist;
    status = list_traverse(varlist, varname, NCF_ListTraverse_FoundVarName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    /*
     * Remove the variable from the dataset list and free it
     * (and its lists) after getting its varid
     */
    var_ptr = (ncvar *) list_remove_curr(varlist, __FILE__, __LINE__);
    ivar = var_ptr->varid;
    ncf_free_variable((char *) var_ptr);

    /* Reset the varids for variables added to external datasets with LET/D
     * For the virtual user-variable dataset, leave varids alone.
     */
    if ( *dset > PDSET_UVARS ) {
        list_mvfront(varlist);
        do {
            var_ptr = (ncvar *) list_curr(varlist);
            if ( (var_ptr != NULL) && (var_ptr->varid > ivar) ) {
                var_ptr->varid = var_ptr->varid - 1;
            }
        } while ( list_mvnext(varlist) != NULL );
    }

    /* Decrement number of variables in the dataset. */
    nc_ptr->nvars = nc_ptr->nvars - 1;

    return FERR_OK;
}

/* ----
 * Initialize new dataset to contain an aggregate dataset
 * save in GLOBAL_ncdsetList for attribute handling
 */
int FORTRAN(ncf_init_agg_dset)( int *setnum, char name[] )
{
    ncdset nc;
    ncvar var;
    ncatt att;

    ncf_init_dataset(&nc);
    strcpy(nc.fername, name);
    nc.fer_dsetnum = *setnum;
    nc.ngatts = 1;
    nc.its_agg = 1;
    nc.num_agg_members = 0;

    /* set up pseudo-variable . the list of variables */
    ncf_init_variable(&var);
    strcpy(var.name, ".");
    var.type = NC_CHAR;
    var.outtype = NC_CHAR;
    var.varid = 0;
    var.natts = 1;
    var.ndims = 0;

    /* set global attribute, aggregate name */
    ncf_init_attribute(&att);
    att.type = NC_CHAR;
    att.outtype = NC_CHAR;
    att.attid = 1;
    strcpy(att.name, "aggregate name" );
    att.len = strlen(name);
    att.string = (char *) FerMem_Malloc((att.len+1)* sizeof(char), __FILE__, __LINE__);
    strcpy(att.string, name);

    /*Save attribute in linked list of attributes for variable .*/
    if ( var.varattlist == NULL ) {
        var.varattlist = list_init(__FILE__, __LINE__);
        if ( var.varattlist == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_agg_dset: Unable to initialize GLOBAL attributes list.\n");
            return -1;
        }
    }
    list_insert_after(var.varattlist, (char *) &att, sizeof(ncatt), __FILE__, __LINE__);

    /* global attributes list complete */

    /* Initialize linked list of variables for this dataset */
    if ( nc.dsetvarlist == NULL ) {
        nc.dsetvarlist = list_init(__FILE__, __LINE__);
        if ( nc.dsetvarlist == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_agg_dset: Unable to initialize variable list.\n");
            return -1;
        }
    }
    list_insert_after(nc.dsetvarlist, (char *) &var, sizeof(ncvar), __FILE__, __LINE__);

    /*Initialize list of aggregate members for this dataset */
    if ( nc.agg_dsetlist == NULL ) {
        nc.agg_dsetlist = list_init(__FILE__, __LINE__);
        if ( nc.agg_dsetlist == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_agg_dset: Unable to initialize aggregate list.\n");
            return -1;
        }
    }

    /* Add dataset to global nc dataset linked list*/
    if ( GLOBAL_ncdsetList == NULL ) {
        GLOBAL_ncdsetList = list_init(__FILE__, __LINE__);
        if ( GLOBAL_ncdsetList == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize GLOBAL_ncDsetList.\n");
            return -1;
        }
    }
    list_insert_after(GLOBAL_ncdsetList, (char *) &nc, sizeof(ncdset), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * Add a new aggregate member to an aggregate dataset.
 */
int FORTRAN(ncf_add_agg_member)( int *dset, int *sequence_number, int *member_dset )
{
    ncdset *nc_ptr;
    ncagg agg;

    /*
     * Get the dataset pointer.
     */
    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of aggregate members. Put the new info at the end.
     * The add_dsetlist should already exist.
     */
    if ( nc_ptr->agg_dsetlist == NULL )
        return ATOM_NOT_FOUND;

    /* Save aggregate member number in linked list of aggregate members for this dataset */
    agg.dsetnum = *member_dset;
    agg.aggSeqNo = *sequence_number;
    list_mvrear(nc_ptr->agg_dsetlist);
    list_insert_after(nc_ptr->agg_dsetlist, (char *) &agg, sizeof(agg), __FILE__, __LINE__);
    nc_ptr->num_agg_members = nc_ptr->num_agg_members + 1;

    return FERR_OK;
}

/* ----
 * Find a dataset based on its integer ID and return the
 * number of aggregate member datasets
 */
int FORTRAN(ncf_get_agg_count)( int *dset, int *num_agg_dsets )
{
    ncdset *nc_ptr;

    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    *num_agg_dsets = nc_ptr->num_agg_members;

    return FERR_OK;
}


/* ----
 * Find a dataset based on its integer ID and for a given member number
 * return the Ferret dataset number
 */
int FORTRAN(ncf_get_agg_member)( int *dset, int *imemb, int *membset )
{
    ncdset *nc_ptr;
    LIST *agglist;
    int status;
    ncagg *agg_ptr;

    nc_ptr = ncf_get_ds_ptr(dset);
    if ( nc_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of aggregation members.
     */
    agglist = nc_ptr->agg_dsetlist;
    if ( agglist == NULL )
        return ATOM_NOT_FOUND;
    status = list_traverse(agglist, (char *) imemb, NCF_ListTraverse_FoundDsMemb, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    agg_ptr = (ncagg *) list_curr(agglist);

    *membset = agg_ptr->dsetnum;

    return FERR_OK;
}


/* ----
 * Add description for variable in aggregate dataset.
 * Given the aggregate dataset, and the varid of the variable, and
 * the aggregate sequence-number, save the variable type (1=file-variable,
 * 3=user-var), the Ferret datset id, the grid, the Ferret line number
 * for the aggregate dimension, and the sequence number in ds_var_code
 * or uvar_name_code.
 */
int FORTRAN(ncf_add_agg_var_info)( int *dset, int *varid, int *imemb,
                                   int *vtype, int *datid, int *igrid, int *iline, int *nv )
{
    ncvar *var_ptr;
    LIST *varagglist;
    ncagg_var_descr vdescr;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of members for the variable in the dataset
     */
    varagglist = var_ptr->varagglist;
    if ( varagglist == NULL )
        return ATOM_NOT_FOUND;

    vdescr.imemb = *imemb;
    vdescr.vtype = *vtype;
    vdescr.datid = *datid;
    vdescr.gnum    = *igrid;
    vdescr.iline = *iline;
    vdescr.nv        = *nv;

    /* Increment number of grid values saved */
    var_ptr->nmemb = var_ptr->nmemb + 1;

    /*Save grid number in linked list of grid for this variable */
    list_insert_after(var_ptr->varagglist, (char *) &vdescr, sizeof(ncagg_var_descr), __FILE__, __LINE__);

    return FERR_OK;
}

/* ----
 * For a variable in aggregate aggregate dataset, store its grid.
 * Given the aggregate dataset, the varid of the variable, and
 * the aggregate sequence-number, save the grid of the variable.
 */
int FORTRAN(ncf_put_agg_memb_grid)( int *dset, int *varid, int *imemb, int *igrid )
{
    ncvar *var_ptr;
    LIST *varagglist;
    int status;
    ncagg_var_descr *vdescr_ptr;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of members for the variable in the dataset. Reset grid number.
     */
    varagglist = var_ptr->varagglist;
    if ( varagglist == NULL )
        return ATOM_NOT_FOUND;
    status = list_traverse(varagglist, (char *) imemb, NCF_ListTraverse_FoundVariMemb, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    vdescr_ptr = (ncagg_var_descr *) list_curr(varagglist);
    vdescr_ptr->gnum = *igrid;

    return FERR_OK;
}

/* ----
 * Given the aggregate dataset, varid, and member number, return the
 * variable type (1=file-variable, 3=user-var), the Ferret datset id,
 * the grid and the sequence number in ds_var_code or uvar_name_code.
 */
int FORTRAN(ncf_get_agg_var_info)( int *dset, int *varid, int *imemb, int* vtype,
                                   int *datid, int *igrid, int *iline, int *nv)
{
    ncvar *var_ptr;
    ncagg_var_descr *vdescr_ptr;
    int status;
    LIST *varagglist;

    /* Get the variable */
    var_ptr = ncf_get_ds_var_ptr(dset, varid);
    if ( var_ptr == NULL )
        return ATOM_NOT_FOUND;

    /*
     * Get the list of aggregate-grids for the variable in the dataset
     */
    varagglist = var_ptr->varagglist;

    status = list_traverse(varagglist, (char *) imemb, NCF_ListTraverse_FoundVariMemb, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    vdescr_ptr = (ncagg_var_descr *) list_curr(varagglist);

    *vtype = vdescr_ptr->vtype;
    *datid = vdescr_ptr->datid;
    *igrid = vdescr_ptr->gnum;
    *iline = vdescr_ptr->iline;
    *nv        = vdescr_ptr->nv;

    return FERR_OK;
}

/* ----
 * For attributes that Ferret always writes, set the output flag to 1
 * All others are not written by default. The flag can be set to 1 by
 * the user.    The modulo flag is set to 0. This will be overriden in
 * the Ferret code depending on the value of the modulo attribute.
 */

static int initialize_output_flag( char *attname, int is_axis )
{
    /* attributes on coordinate variables */
    if ( strcmp(attname, "axis") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "units") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "calendar") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "positive") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "point_spacing") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "modulo") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "time_origin") == 0 ) {
        return 1;
    }
    /* attributes on variables */
    if ( strcmp(attname, "missing_value") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "_FillValue") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "long_name") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "title") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "history") == 0 ) {
        return 1;
    }
    if ( strcmp(attname, "bounds") == 0 ) {
        return 1;
    }
    /* write scale attributes on non-coordinate variables */
    if ( is_axis == 0 ) {
        if ( strcmp(attname, "scale_factor") == 0 ) {
            return 1;
        }
        if ( strcmp(attname, "add_offset") == 0 ) {
            return 1;
        }
    }
    return 0;
}


/* *******************************
 *    uvar grid management routines
 * *******************************
 */


/* ----
 * Find variable based on its variable ID and LIST_dset ID
 * Free ("purge" in Ferret-speak) the entire list of uvar grids
 */
int FORTRAN(ncf_free_uvar_grid_list)( int *LIST_dset, int *uvarid )
{
    LIST *varlist;
    int status;
    ncvar *var_ptr;
    LIST *uvgridList;
    uvarGrid *uvgrid_ptr;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    varlist = ncf_get_ds_varlist(LIST_dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;

    /* find the relevant LET var (i.e. uvar) */
    status = list_traverse(varlist, (char *) uvarid, NCF_ListTraverse_FoundUvarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    var_ptr = (ncvar *) list_curr(varlist);

    /* remove all elements from the uvar grid list but do not remove the list itself */
    uvgridList = var_ptr->uvarGridList;
    if ( uvgridList != NULL ) {
        while ( ! list_empty(uvgridList) ) {
            uvgrid_ptr = (uvarGrid *)list_remove_front(uvgridList, __FILE__, __LINE__);
            /* paranoia */
            memset(uvgrid_ptr, 0, sizeof(uvarGrid));
            FerMem_Free(uvgrid_ptr, __FILE__, __LINE__);
        }
    }

    return FERR_OK;
}


/* ----
 * Find variable based on its variable ID and LIST_dset ID
 * return the grid corresponding to the ith element in the grid list
 */
int FORTRAN(ncf_next_uvar_grid_in_list)( int *LIST_dset, int *uvarid, int *ith, int *grid )
{
    LIST *varlist;
    int status;
    ncvar *var_ptr;
    LIST *uvgridList;
    uvarGrid *uvgrid_ptr;
    int i;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    varlist = ncf_get_ds_varlist(LIST_dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;

    /* find the relevant LET var (i.e. uvar) */
    status = list_traverse(varlist, (char *) uvarid, NCF_ListTraverse_FoundUvarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    var_ptr = (ncvar *) list_curr(varlist);

    /* remove all elements from the uvar grid list but do not remove the list itself */
    uvgridList = var_ptr->uvarGridList;
    if ( uvgridList != NULL ) {


    /*
     * Return the ith grid
     */
    list_mvfront(uvgridList);
    for (i = 0; i < *ith; i++) {
        uvgrid_ptr = (uvarGrid *) list_curr(uvgridList);
        *grid = uvgrid_ptr->grid;
        list_mvnext(uvgridList); 
        } 

    }

    return FERR_OK;
}



/* ----
 * Find variable based on its variable ID and LIST_dset ID
 * Store a grid/context_dset pair for the variable
 *
 * The dual dataset arguments arise because Ferret's global uvars are managed
 * in the c LIST structures as a special dataset -- PDSET_UVARS
 * By contrast LET/D uvars are managed in the c LIST structure of the parent dataset
 * So we refer to the dataset that owns (parents) the uvar as LIST_dset
 * and we refer to the dataset in which Ferret is evaluating the uvar is as context_dset
 */
int FORTRAN(ncf_set_uvar_grid)( int *LIST_dset, int *varid, int *grid, int *datatype, int *context_dset )
{
    LIST *varlist;
    int status;
    ncvar *var_ptr;
    LIST *uvgridlist;
    uvarGrid *uvgrid_ptr;
    uvarGrid uvgrid;
    int uvgrid_list_len;
    int i;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    varlist = ncf_get_ds_varlist(LIST_dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;
    status = list_traverse(varlist, (char *) varid, NCF_ListTraverse_FoundUvarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    var_ptr=(ncvar *)list_curr(varlist);

    /*
     * if a grid already exists for this context dataset
     * remove it before continuing
     */
    uvgridlist = var_ptr->uvarGridList;
    if ( uvgridlist == NULL )
        return ATOM_NOT_FOUND;
    status = list_traverse(uvgridlist, (char *) context_dset, NCF_ListTraverse_FoundGridDset, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status == LIST_OK ) {
        uvgrid_ptr = (uvarGrid *) list_remove_curr(uvgridlist, __FILE__, __LINE__);
        /* paranoia */
        memset(uvgrid_ptr, 0, sizeof(uvarGrid));
        FerMem_Free(uvgrid_ptr, __FILE__, __LINE__);
        uvgrid_ptr = NULL;
    }

    /*
     * Fill the uvarGrid structure
     */
    uvgrid.grid  = *grid;
    uvgrid.dset  = *context_dset;
    uvgrid.dtype = *datatype;

    /*
     * Set the auxiliary variables as unspecified at this point
     */
    for (i = 0; i < NFERDIMS; i++) {
        uvgrid.auxCat[i] = 0;
        uvgrid.auxVar[i] = 0;
    }

    /*
     * Save it in the grid list of this uvar
     */
    list_insert_after(uvgridlist, (char *) &uvgrid, sizeof(uvarGrid), __FILE__, __LINE__);

    uvgrid_list_len = (int) list_size(uvgridlist);
    if (uvgrid_list_len > 1) { 
        i = 1;
    }

    return FERR_OK;
}

/* ----
 * Find variable based on its variable ID and LIST_dset ID
 * Return the grid and variable datatype that corresponds to the context_dset pair
 *
 * The dual dataset arguments arise because Ferret's global uvars are managed
 * in the c LIST structures as a special dataset -- PDSET_UVARS
 * By contrast LET/D uvars are managed in the c LIST structure of the parent dataset
 * So we refer to the dataset that owns (parents) the uvar as LIST_dset
 * and we refer to the dataset in which Ferret is evaluating the uvar is as context_dset
 */
int FORTRAN(ncf_get_uvar_grid)( int *LIST_dset, int *uvarid, int *context_dset, int *uvgrid, int *uvdtype )
{
    LIST *varlist;
    int status;
    ncvar *var_ptr;
    LIST *uvgridlist;
    uvarGrid *uvgrid_ptr;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    varlist = ncf_get_ds_varlist(LIST_dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;

    /* find the relevant LET var (i.e. uvar) */
    status = list_traverse(varlist, (char *) uvarid, NCF_ListTraverse_FoundUvarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    var_ptr = (ncvar *) list_curr(varlist);

    /* find the relevant grid/dataset pair owned by this uvar */
    uvgridlist = var_ptr->uvarGridList;
    if ( uvgridlist == NULL )
        return ATOM_NOT_FOUND;

    status = list_traverse(uvgridlist, (char *) context_dset, NCF_ListTraverse_FoundGridDset, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    uvgrid_ptr = (uvarGrid *) list_curr(uvgridlist);
    *uvgrid = uvgrid_ptr->grid;
    *uvdtype = uvgrid_ptr->dtype;

    return FERR_OK;
}

/* ----
 * Find variable based on its variable ID and LIST_dset ID
 * Store a grid/context_dset pair for the variable
 *
 * The dual dataset arguments arise because Ferret's global uvars are managed
 * in the c LIST structures as a special dataset -- PDSET_UVARS
 * By contrast LET/D uvars are managed in the c LIST structure of the parent dataset
 * So we refer to the dataset that owns (parents) the uvar as LIST_dset
 * and we refer to the dataset in which Ferret is evaluating the uvar is as context_dset
 */
int FORTRAN(ncf_set_uvar_aux_info)( int *LIST_dset, int *varid, int aux_cat[], int aux_var[], int *context_dset )
{
    ncvar *var_ptr;
    int status;
    LIST *varlist;
    LIST *uvgridlist;
    uvarGrid *uvgrid;
    int i;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    varlist = ncf_get_ds_varlist(LIST_dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;

    status = list_traverse(varlist, (char *) varid, NCF_ListTraverse_FoundUvarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    var_ptr = (ncvar *) list_curr(varlist);

    /*
     * a grid must already exist for this context dataset
     */
    uvgridlist = var_ptr->uvarGridList;
    if ( uvgridlist == NULL )
        return ATOM_NOT_FOUND;
    status = list_traverse(uvgridlist, (char *) context_dset, NCF_ListTraverse_FoundGridDset, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    uvgrid = (uvarGrid *) list_curr(uvgridlist);

    /*
     * Fill the uvar aux arrays
     */
    for (i = 0; i < NFERDIMS; i++) {
        uvgrid->auxCat[i] = aux_cat[i];
        uvgrid->auxVar[i] = aux_var[i];
    }

    return FERR_OK;
}

/* ----
 * Find variable based on its variable ID and LIST_dset ID
 * Store a grid/context_dset pair for the variable
 *
 * The dual dataset arguments arise because Ferret's global uvars are managed
 * in the c LIST structures as a special dataset -- PDSET_UVARS
 * By contrast LET/D uvars are managed in the c LIST structure of the parent dataset
 * So we refer to the dataset that owns (parents) the uvar as LIST_dset
 * and we refer to the dataset in which Ferret is evaluating the uvar is as context_dset
 */
int FORTRAN(ncf_get_uvar_aux_info)( int *LIST_dset, int *varid, int *context_dset,
                                    int aux_cat[], int aux_var[] )
{
    ncvar *var_ptr;
    int status;
    LIST *varlist;
    LIST *uvgridlist;
    uvarGrid *uvgrid;
    int i;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    varlist = ncf_get_ds_varlist(LIST_dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;

    status = list_traverse(varlist, (char *) varid, NCF_ListTraverse_FoundUvarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    var_ptr = (ncvar *) list_curr(varlist);

    /*
     * a grid must already exists for this context dataset
     */
    uvgridlist = var_ptr->uvarGridList;
    if ( uvgridlist == NULL )
        return ATOM_NOT_FOUND;
    status = list_traverse(uvgridlist, (char *) context_dset, NCF_ListTraverse_FoundGridDset, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    uvgrid = (uvarGrid *) list_curr(uvgridlist);

    /*
     * Return the uvar aux arrays
     */
    for (i = 0; i < NFERDIMS; i++) {
        aux_cat[i] = uvgrid->auxCat[i];
        aux_var[i] = uvgrid->auxVar[i];
    }

    return FERR_OK;
}

/* ----
 * Find variable based on its variable ID and LIST_dset ID
 * Return the length of the LIST of saved grids
 *
 * The dual dataset arguments arise because Ferret's global uvars are managed
 * in the c LIST structures as a special dataset -- PDSET_UVARS
 * By contrast LET/D uvars are managed in the c LIST structure of the parent dataset
 * So we refer to the dataset that owns (parents) the uvar as LIST_dset
 * and we refer to the dataset in which Ferret is evaluating the uvar is as context_dset
 */
int FORTRAN(ncf_get_uvar_grid_list_len)( int *LIST_dset, int *uvarid, int *uvgrid_list_len )
{
    ncvar *var_ptr;
    int status;
    LIST *varlist;
    LIST *uvgridlist;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    varlist = ncf_get_ds_varlist(LIST_dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;

    /* find the relevant LET var (i.e. uvar) */
    status = list_traverse(varlist, (char *) uvarid, NCF_ListTraverse_FoundUvarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    var_ptr = (ncvar *) list_curr(varlist);

    uvgridlist = var_ptr->uvarGridList;
    if ( uvgridlist == NULL )
        return ATOM_NOT_FOUND;
    *uvgrid_list_len = (int) list_size(uvgridlist);

    return FERR_OK;
}

/* ----
 * Find variable based on its variable ID and LIST_dset ID
 * Delete the grid that corresponds to the context_dset
 * from the uvarGridList
 */
int FORTRAN(ncf_delete_uvar_grid)( int *LIST_dset, int *uvarid, int *context_dset )
{
    LIST *varlist;
    int status;
    ncvar *var_ptr;
    LIST *uvgridlist;
    uvarGrid *uvgrid_ptr;

    /*
     * Get the list of variables, find pointer to variable varid.
     */
    varlist = ncf_get_ds_varlist(LIST_dset);
    if ( varlist == NULL )
        return ATOM_NOT_FOUND;

    /* find the relevant LET var (i.e. uvar) */
    status = list_traverse(varlist, (char *) uvarid, NCF_ListTraverse_FoundUvarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;
    var_ptr = (ncvar *) list_curr(varlist);

    /* find the relevant grid/dataset pair owned by this uvar */
    uvgridlist = var_ptr->uvarGridList;
    if ( uvgridlist == NULL )
        return ATOM_NOT_FOUND;

    status = list_traverse(uvgridlist, (char *) context_dset, NCF_ListTraverse_FoundGridDset, (LIST_FRNT | LIST_FORW | LIST_ALTR));
    if ( status != LIST_OK )
        return ATOM_NOT_FOUND;

    /* Remove this grid from uvaGridList list */
    uvgrid_ptr = (uvarGrid *) list_remove_curr(uvgridlist, __FILE__, __LINE__);
    /* paranoia */
    memset(uvgrid_ptr, 0, sizeof(uvarGrid));
    FerMem_Free(uvgrid_ptr, __FILE__, __LINE__);

    return FERR_OK;
}


/* ***********************************
 *    search routines for LIST traversal
 * ***********************************
 * Note that a return value of zero means a match was found
 * (as with strcmp).    Any non-zero value indicated not a match,
 * and sign does not matter.
 */

/* ----
 * See if the name in data matches the ferret dset name in
 * curr. Ferret always capitalizes everything so be case INsensitive.
 */
static int NCF_ListTraverse_FoundDsetName( char *data, char *curr )
{
    ncdset *nc_ptr = (ncdset *) curr;

    return strcasecmp(data, nc_ptr->fername);
}

/* ----
 * See if the dataset id in data matches the ferret dset id in curr.
 */
static int NCF_ListTraverse_FoundDsetID( char *data, char *curr )
{
    ncdset *nc_ptr = (ncdset *) curr;
    int ID = *((int *) data);

    if ( ID == nc_ptr->fer_dsetnum ) {
        return 0; /* found match */
    }
    return 1;
}

/* ----
 * See if the name in data matches the variable name in
 * curr. Ferret always capitalizes everything so be case INsensitive,
 * unless the string has been passed in inside single quotes.
 */
static int NCF_ListTraverse_FoundVarName( char *data, char *curr )
{
    ncvar *var_ptr = (ncvar*) curr;

    return strcasecmp(data, var_ptr->name);
}

/* ----
 * See if the name in data matches the variable name in
 * curr. Make the string comparison case-sensive.
 */
static int NCF_ListTraverse_FoundVarNameCase( char *data, char *curr )
{
    ncvar *var_ptr = (ncvar*) curr;

    return strcmp(data, var_ptr->name);
}

/* ----
 * See if the ID in data matches the variable ID in curr.
 */
static int NCF_ListTraverse_FoundVarID( char *data, char *curr )
{
    ncvar *var_ptr = (ncvar*) curr;
    int ID = *((int *) data);

    if ( ID == var_ptr->varid)    {
        return 0; /* found match */
    }
    return 1;
}

/* ----
 * See if the ID in data matches the uvar ID in curr.
 */
static int NCF_ListTraverse_FoundUvarID( char *data, char *curr )
{
    ncvar *var_ptr = (ncvar*) curr;
    int ID = *((int *) data);

    if ( ID == var_ptr->uvarid ) {
        return 0; /* found match */
    }
    return 1;
}

/* ----
 * See if the name in data matches the attribute name in curr.
 */
static int NCF_ListTraverse_FoundVarAttName( char *data, char *curr )
{
    ncatt *att_ptr = (ncatt *) curr;

    return strcasecmp(data, att_ptr->name);
}

/* ----
 * See if the name in data matches the attribute name in curr.
 * Make the string comparison case-sensive.
 */
static int NCF_ListTraverse_FoundVarAttNameCase( char *data, char *curr )
{
    ncatt *att_ptr = (ncatt *) curr;

    return strcmp(data, att_ptr->name);
}

/* ----
 * See if there is an ID in data matches the attribute id in curr.
 */
static int NCF_ListTraverse_FoundVarAttID( char *data, char *curr )
{
    ncatt *att_ptr = (ncatt *) curr;
    int ID = *((int *) data);

    if ( ID == att_ptr->attid ) {
        return 0; /* found match */
    }
    return 1;
}

/* ----
 * See if there is an ID in data matches the dset-member id.
 */
static int NCF_ListTraverse_FoundVariMemb( char *data, char *curr )
{
    ncagg_var_descr *vdescr_ptr = (ncagg_var_descr *) curr;
    int ID = *((int *) data);

    if ( ID == vdescr_ptr->imemb ) {
        return 0; /* found match */
    }
    return 1;
}


/* ----
 * See if there is a match on the dset sequence number.
 */
static int NCF_ListTraverse_FoundDsMemb( char *data, char *curr )
{
    ncagg *agg_ptr = (ncagg *) curr;
    int ID = *((int *) data);

    /* 12/15 -- search is successful if sequence number (FORTRAN index) matches */
    if ( ID == agg_ptr->aggSeqNo ) {
        return 0; /* found match */
    }
    return 1;
}

/* ----
 * See if there is a match on the context dset
 */
static int NCF_ListTraverse_FoundGridDset( char *data, char *curr )
{

    uvarGrid *uvgrid_ptr = (uvarGrid *) curr;
    int ID = *((int *) data);

    if ( ID == uvgrid_ptr->dset ) {
        return 0; /* found match */
    }
    return 1;
}

