/*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granteHd the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
*
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

#include <wchar.h>
#include <unistd.h>		/* for convenience */
#include <stdlib.h>		/* for convenience */
#include <stdio.h>		/* for convenience */
#include <string.h>		/* for convenience */
#include <fcntl.h>		/* for fcntl() */
#include <assert.h>
#include <sys/types.h>	        /* required for some of our prototypes */
#include <sys/stat.h>
#include <sys/errno.h>

#include "netcdf.h"
#include "nc.h"
#include "list.h"  /* locally added list library */
#include "NCF_Util.h"

/* ................ Global Variables ................ */

static LIST  *GLOBAL_ncdsetList;
static int list_initialized = FALSE;

/* ............. Function Declarations .............. */
/*
 * Note that all routines called directly from Ferret,
 * ie. directly from Fortran, should be all lower case,
 * be of type 'void', pass by reference and should end with 
 * an underscore.
 */


/* .... Functions called by Ferret .... */

int  FORTRAN(ncf_inq_ds)( int *, int *, int *, int *, int *);
int  FORTRAN(ncf_inq_ds_dims)( int *, int *, char *, int *, int *);
int  FORTRAN(ncf_inq_var) (int *, int *, char *, int *, int *, int *, int *, int *, int *, int * );

int  FORTRAN(ncf_inq_var_att)( int *, int *, int *, char *, int *, int *, int *, int *);

int  FORTRAN(ncf_get_dsnum)( char * );
int  FORTRAN(ncf_get_dsname)( int *, char *);
int  FORTRAN(ncf_get_dim_id)( int *, char *);

int  FORTRAN(ncf_get_var_name)( int *, int *, char *);
int  FORTRAN(ncf_get_var_id)( int *, int*, char *); 
int  FORTRAN(ncf_get_var_id_case)( int *, int*, char *); 
int  FORTRAN(ncf_get_var_axflag)( int *, int *, int *, int *); 
int  FORTRAN(ncf_get_var_seq)( int *, char *); 
int  FORTRAN(ncf_get_var_attr_name) (int *, int *, int *, int *, char*);
int  FORTRAN(ncf_get_var_attr_id) (int *, int *, char* , int*);
int  FORTRAN(ncf_get_var_attr_id_case) (int *, int *, char* , int*);
int  FORTRAN(ncf_get_var_attr) (int *, int *, char* , char* , int *, double *);
int  FORTRAN(ncf_get_var_outflag) (int *, int *, int *);
int  FORTRAN(ncf_get_var_outtype) (int *, int *,  int *);

int  FORTRAN(ncf_init_uvar_dset)( int *);
int  FORTRAN(ncf_add_dset)( int *, int *, char *, char *);
int  FORTRAN(ncf_init_other_dset)( int *, char *, char *);
int  FORTRAN(ncf_delete_dset)( int *);
int  FORTRAN(ncf_delete_var_att)( int *, int *, char *);
int  FORTRAN(ncf_delete_var)( int *, char *);

int  FORTRAN(ncf_add_var)( int *, int *, int *, char *, char *, char *, double *);

int  FORTRAN(ncf_add_var_num_att)( int *, int *, char *, int *, int *, int *, float *);
int  FORTRAN(ncf_add_var_str_att)( int *, int *, char *, int *, int *, int *, char *);

int  FORTRAN(ncf_repl_var_att)( int *, int *, char *, int *, int *, float *, char *);
int  FORTRAN(ncf_set_att_flag)( int *, int *, char *, int *);
int  FORTRAN(ncf_set_var_out_flag)( int *, int *, int *);
int  FORTRAN(ncf_set_var_outtype)( int *, int *, int *);
int  FORTRAN(ncf_set_axdir)(int *, int *, int *);
int  FORTRAN(ncf_transfer_att)(int *, int *, int *, int *, int *);
 
/* .... Functions called internally .... */

ncdset *ncf_ptr_from_dset(int *);
LIST *ncf_get_ds_varlist( int *);
LIST *ncf_get_ds_var_attlist (int *, int *);

int initialize_output_flag (char *);
int NCF_ListTraverse_FoundDsetName( char *, char * );
int NCF_ListTraverse_FoundDsetID( char *, char * );
int NCF_ListTraverse_FoundVarName( char *, char * );
int NCF_ListTraverse_FoundVarNameCase( char *, char * );
int NCF_ListTraverse_FoundVarID( char *, char * );
int NCF_ListTraverse_FoundVarAttName( char *, char * );
int NCF_ListTraverse_FoundVarAttNameCase( char *, char * );
int NCF_ListTraverse_FoundVarAttID( char *, char * );

void list_free(LIST *, int ); 

/*
 * Find a dataset based on its integer ID and return the scalar information:
 * ndims, nvars, ngatts, recdim.
 */

int FORTRAN(ncf_inq_ds)( int *dset, int *ndims, int *nvars, int *ngatts, int *recdim )
{
  ncdset *nc_ptr=NULL;
  int return_val;

  return_val = ATOM_NOT_FOUND;
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL ) { return return_val; }

  *ndims = nc_ptr->ndims;
  *nvars = nc_ptr->nvars;
  *ngatts = nc_ptr->ngatts;
  *recdim = nc_ptr->recdim;

  return_val = FERR_OK; 
  return return_val; 
}

/* ----
 * Find a dataset based on its integer ID and return the dimension info for
 * dimension given.
 */
int  FORTRAN(ncf_inq_ds_dims)( int *dset, int *idim, char dname[], int *namelen, int *dimsize)
{
  ncdset *nc_ptr=NULL;
  int return_val;

  return_val = ATOM_NOT_FOUND;
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL ) { return return_val; }
  
  strcpy (dname, nc_ptr->dims[*idim-1].name);
  *namelen = strlen(dname);
  *dimsize = nc_ptr->dims[*idim-1].size;

  return_val = FERR_OK; 
  return return_val; 
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and 
 * variable id. Return the variable name (in its original upper/lower 
   case form), type, nvdims, vdims, nvatts.
 */

 int FORTRAN(ncf_inq_var) (int *dset, int *varid, char vname[], int *len_vname, int *vtype, int *nvdims,
     int *nvatts, int* coord_var, int *outflag, int *vdims)

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  int i;
  int ndx;
  int the_dim;
  int outdims;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;

  return_val = ATOM_NOT_FOUND;  
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);
  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }
  
  var_ptr=(ncvar *)list_curr(varlist); 

  strcpy(vname, var_ptr->name);
  *len_vname = strlen(vname);
  *vtype = var_ptr->type;
  *nvdims = var_ptr->ndims;
  *nvatts = var_ptr->natts;
  *outflag = var_ptr->all_outflag;
  *coord_var = var_ptr->is_axis;
   
   for (i=0; i <var_ptr->ndims ;i++ )
  {
	  the_dim =  var_ptr->dims[i];
	  vdims[i] = the_dim ;
  }

  return_val = FERR_OK;
  return return_val;
}


/* ----
 * Find a variable in a dataset based on the dataset integer ID and 
 * variable id. Return the variable output type.
 */

 int FORTRAN(ncf_get_var_outtype) (int *dset, int *varid,  int *outtype)

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;


  return_val = ATOM_NOT_FOUND;

  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

  if (*varid > nc_ptr->nvars+1) return return_val;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);
  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }
  
  var_ptr=(ncvar *)list_curr(varlist); 

  *outtype = var_ptr->outtype;
   
  return_val = FERR_OK;
  return return_val;
}
 

/* ----
 * Find a variable attribute based on its variable ID and dataset ID, and attribute name
 * Return the attribute name, type, length, and output flag
 */
int  FORTRAN(ncf_inq_var_att)( int *dset, int *varid, int *attid, char attname[], int *namelen, int *attype, int *attlen, int *attoutflag)

{
  ncatt *att_ptr=NULL;
  ncvar *var_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  if (var_ptr->natts < 1) return ATOM_NOT_FOUND;

   /*
   * Get the list of attributes for the variable in the dataset
   */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  status = list_traverse(varattlist, attid, NCF_ListTraverse_FoundVarAttID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
    }

  att_ptr=(ncatt *)list_curr(varattlist); 

  strcpy(attname, att_ptr->name);
  *namelen = strlen(attname);
  *attype = att_ptr->type; 
  *attlen = att_ptr->len;
  *attoutflag = att_ptr->outflag;

  return_val = FERR_OK;
  return return_val;
}

/* ----
 * Find a dataset based on its name and
 * return the ferret dataset number.
 */

int FORTRAN(ncf_get_dsnum)( char name[] )
{
  ncdset *nc_ptr=NULL;
  int status=LIST_OK;

  static int return_val=0; /* static because it needs to exist after the return statement */

  /*
   * Find the dataset.
   */

  status = list_traverse(GLOBAL_ncdsetList, name, NCF_ListTraverse_FoundDsetName, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, set the dset to ATOM_NOT_FOUND.
   */
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }

  nc_ptr=(ncdset *)list_curr(GLOBAL_ncdsetList); 

  return_val = nc_ptr->fer_dsetnum;
  return return_val;
}

/* ----
 * Find a dataset based on its integer ID and return the name.
 */

int FORTRAN(ncf_get_dsname)( int *dset, char name[] )
{
  ncdset *nc_ptr=NULL;
  int return_val;

  return_val = ATOM_NOT_FOUND;
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL ) { return return_val; }

  strcpy(name, nc_ptr->fername);

  return_val = FERR_OK;
  return return_val;
}

/* ----
 * Find a dataset based on its integer ID and a dimension name. Return the dimension ID.
 */
int FORTRAN(ncf_get_dim_id)( int *dset, char dname[])
{
  ncdset *nc_ptr=NULL;
  int return_val;
  int idim;
  int sz;
  int szdim;

  return_val = ATOM_NOT_FOUND;
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL ) { return return_val; }
  
  for (idim = 0; idim < nc_ptr->ndims; idim++) {
	sz = strlen(dname);
	szdim = strlen(nc_ptr->dims[idim].name);
    if ( (sz == szdim) &&
		 (nc_ptr->dims[idim].size !=0) && 
		 (strncmp(dname, nc_ptr->dims[idim].name, sz) == 0) )
    { return_val = idim + 1;
	  return return_val;
    } 
  }
  return return_val;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and 
 * variable id. Return the variable name.
 */
 int FORTRAN(ncf_get_var_name) (int *dset, int* ivar, char* string)

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  int i;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;
  LIST *dummy;

  return_val = ATOM_NOT_FOUND;  
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

  if (*ivar > nc_ptr->nvars) return return_val;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);
  dummy = list_mvfront(varlist);
  var_ptr=(ncvar *)list_front(varlist); 

  for (i = 0; i < *ivar; i++) {
     strcpy(string, var_ptr->name); 
     dummy = list_mvnext(varlist);
     var_ptr=(ncvar *)list_curr(varlist);  
  }
  
  free(dummy);
  return_val = FERR_OK;
  return return_val;
}

/* ----
 * Find a variable in a dataset based on the dataset integer ID and 
 * variable name. Return the variable id, or NOT FOUND if it does not exist
 */
 int FORTRAN(ncf_get_var_id) (int *dset, int *varid, char string[])

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;

  return_val = ATOM_NOT_FOUND;  
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

   /*
   * Get the list of variables.  
   */

  varlist = ncf_get_ds_varlist(dset);
  status = list_traverse(varlist, string, NCF_ListTraverse_FoundVarName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }
  
  var_ptr=(ncvar *)list_curr(varlist); 
  *varid = var_ptr->varid;
  return_val = FERR_OK;

  return return_val;
}
/* ----
 * Find a variable in a dataset based on the dataset integer ID and 
 * variable name. Return the variable id, or NOT FOUND if it does not exist
 */
 int FORTRAN(ncf_get_var_id_case) (int *dset, int *varid, char string[])

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;

  return_val = ATOM_NOT_FOUND;  
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

   /*
   * Get the list of variables.  
   */

  varlist = ncf_get_ds_varlist(dset);
  status = list_traverse(varlist, string, NCF_ListTraverse_FoundVarNameCase, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }
  
  var_ptr=(ncvar *)list_curr(varlist); 
  *varid = var_ptr->varid;
  return_val = FERR_OK;

  return return_val;
}
/* ----
 * Find a variable in a dataset based on the dataset integer ID and 
 * variable ID. Return the coordinate-axis flag.
 */
 int FORTRAN(ncf_get_var_axflag) (int *dset, int *varid, int* coord_var, int* ax_dir)

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  int i;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;

  return_val = FALSE;  
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);
  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }
  
  var_ptr=(ncvar *)list_curr(varlist); 
  
  *coord_var = var_ptr->is_axis;
  *ax_dir = var_ptr->axis_dir;

  return_val = FERR_OK;
  return return_val;
}


/* ----
 * Find a variable in a dataset based on the dataset integer ID and 
 * variable ID. Return the variable all_outflag for attributes
 */
 int FORTRAN(ncf_get_var_outflag) (int *dset, int *varid, int *iflag)

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  int i;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;

  return_val = 0;  
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

   /*
   * Get the list of variables and the variable based on its id
   */
  varlist = ncf_get_ds_varlist(dset);
  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }
  
  var_ptr=(ncvar *)list_curr(varlist); 
  *iflag = var_ptr->all_outflag;

  return FERR_OK;
}


/* ----
 * Find a variable in a dataset based on the dataset integer ID and 
 * variable name. Return the variable sequence number: variable count 
 * without the coordinate variables.
 */
 int FORTRAN(ncf_get_var_seq) (int *dset, char string[])

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  int i;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;
  LIST *dummy;
  int seq;
  int ivar;

  return_val = ATOM_NOT_FOUND;  
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

   /*
   * Get the list of variables. Make sure varname is in the dataset.
   */
  varlist = ncf_get_ds_varlist(dset);
  status = list_traverse(varlist, string, NCF_ListTraverse_FoundVarName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }
  
  var_ptr=(ncvar *)list_curr(varlist); 
  ivar = var_ptr->varid;

  /* 
   *Move through all the variables up to varid, counting non-coordinate ones.
   */

  dummy = list_mvfront(varlist);
  seq = 0;
  var_ptr=(ncvar *)list_front(varlist); 
  for (i = 0; i < ivar; i++)
  { if (var_ptr->is_axis != TRUE) 
	    {  seq = seq + 1;
        }
     dummy = list_mvnext(varlist);
     var_ptr=(ncvar *)list_curr(varlist);  
  }

  free(dummy);
  return seq;
}

/* ----
 * Find a variable attribute based on the dataset ID and variable ID and attribute name
 * Return the attribute ID
 */
 int FORTRAN(ncf_get_var_attr_id) (int *dset, int *varid, char* attname, int* attid)

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  ncatt *att_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;
  LIST *dummy;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  if (var_ptr->natts < 1) return ATOM_NOT_FOUND;

   /*
   * Get the list of attributes for the variable in the dataset. find attname.
   */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
    }

  att_ptr=(ncatt *)list_curr(varattlist); 
  *attid = att_ptr->attid;

  return_val = FERR_OK;
  return return_val;
}

/* ----
 * Find a variable attribute based on the dataset ID and variable ID and attribute name
 * Return the attribute ID
 */
 int FORTRAN(ncf_get_var_attr_id_case) (int *dset, int *varid, char* attname, int* attid)

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  ncatt *att_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;
  LIST *dummy;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  if (var_ptr->natts < 1) return ATOM_NOT_FOUND;

   /*
   * Get the list of attributes for the variable in the dataset. find attname.
   */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttNameCase, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
    }

  att_ptr=(ncatt *)list_curr(varattlist); 
  *attid = att_ptr->attid;

  return_val = FERR_OK;
  return return_val;
}

/* ----
 * Find a variable attribute based on the dataset ID and variable ID and attribute ID.
 * Return the attribute name.
 */
 int FORTRAN(ncf_get_var_attr_name) (int *dset, int *varid, int* attid, int *namelen, char* attname)

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  ncatt *att_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;
  LIST *dummy;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  if (var_ptr->natts < 1) return ATOM_NOT_FOUND;

   /*
   * Get the list of attributes for the variable in the dataset
   */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  dummy = list_mvfront(varattlist);
  att_ptr=(ncatt *)list_front(varattlist); 

  for (i = 0; i < *attid; i++) {
     strcpy(attname, att_ptr->name);
	 dummy = list_mvnext(varattlist);
     att_ptr=(ncatt *)list_curr(varattlist);  
  }
  
  
  *namelen = strlen(attname);
  return_val = FERR_OK;
  return return_val;
}

/*----
 * Find a variable attribute based on the dataset ID and variable ID and attribute name.
 * Return the attribute, len, and its string or numeric value.
 */
 int FORTRAN(ncf_get_var_attr) (int *dset, int *varid, char* attname, char* string, int *len, double* val)

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  ncatt *att_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  if (var_ptr->natts < 1) return ATOM_NOT_FOUND;

   /*
   * Get the list of attributes for the variable in the dataset
   */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
    }
  
  strcpy(string, "");
  val[0] = NC_FILL_DOUBLE;

  att_ptr=(ncatt *)list_curr(varattlist); 

  *len = att_ptr->len;
  if (att_ptr->type == NC_CHAR)
  { 

	  strcpy(string, att_ptr->string); 
  }
  else 
  { for (i = 0; i < att_ptr->len; i++) {
	  val[i] = att_ptr->vals[i]; }
  }
  return_val = FERR_OK;
  return return_val;
}

/* ----
 * Initialize new dataset to contain user variables and 
 * save in GLOBAL_ncdsetList for attribute handling 
 */

int FORTRAN(ncf_init_uvar_dset)(int *setnum)

{
  ncdset nc; 
  static int return_val=FERR_OK; /* static because it needs to exist after the return statement */
  
    int i;				/* loop controls */
	int ia;
	int iv;
    int nc_status;		/* return from netcdf calls */
    ncatt att;			/* attribute */
    ncvar var;			/* variable */

    strcpy(nc.fername, "UserVariables");
    strcpy(nc.fullpath, " ");
    nc.fer_dsetnum = *setnum;

    nc.ngatts = 1;
    nc.nvars = 0;
	nc.recdim = -1;   /* never used, but initialize anyway*/
	nc.ndims = 4;     /* never used, but initialize anyway*/
    nc.vars_list_initialized = FALSE;

   /* set one global attribute, treat as pseudo-variable . the list of variables */

       strcpy(var.name, ".");

       var.attrs_list_initialized = FALSE;

       var.type = NC_CHAR;
       var.outtype = NC_CHAR;
       var.varid = 0;
	   var.natts = nc.ngatts;
       var.has_fillval = FALSE;
       var.fillval = NC_FILL_FLOAT;
	   var.all_outflag = 1;
	   var.is_axis = FALSE;
	   var.axis_dir = 0;

	   var.attrs_list_initialized = FALSE; 

		  att.outflag = 1;
          att.type = NC_CHAR;
          att.outtype = NC_CHAR;
		  att.len = 21;
          strcpy(att.name, "FerretUserVariables" );
          

      /*Save attribute in linked list of attributes for variable .*/	
       if (!var.attrs_list_initialized) {
          if ( (var.varattlist = list_init()) == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize GLOBAL attributes list.\n");
            return_val = -1;
            return return_val; 
          }
          var.attrs_list_initialized = TRUE;
	  }

       list_insert_after(var.varattlist, &att, sizeof(ncatt));

       /* global attributes list complete */

      /*Save variable in linked list of variables for this dataset */	
       if (!nc.vars_list_initialized) {
          if ( (nc.dsetvarlist = list_init()) == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize variable list.\n");
            return_val = -1;
            return return_val; 
          }
          nc.vars_list_initialized = TRUE;
        }

       list_insert_after(nc.dsetvarlist, &var, sizeof(ncvar));

/* Add dataset to global nc dataset linked list*/ 
  if (!list_initialized) {
    if ( (GLOBAL_ncdsetList = list_init()) == NULL ) {
      fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize GLOBAL_ncDsetList.\n");
      return_val = -1;
      return return_val; 
	}
    list_initialized = TRUE;
  }

  list_insert_after(GLOBAL_ncdsetList, &nc, sizeof(ncdset));
  return_val = FERR_OK;
  return return_val;
  }

/* ----
 * Get file info for a dataset and save in GLOBAL_ncdsetList for attribute handling 
 */

int FORTRAN(ncf_add_dset)(int *ncid, int *setnum, char name[], char path[])

{
  ncdset nc; 
  static int return_val=FERR_OK; /* static because it needs to exist after the return statement */

/* code lifted liberally from ncdump.c Calls in nc library.*/

	char fillc;
    int i;				/* loop controls */
	int ia;
	int iv;
    int nc_status;		/* return from netcdf calls */
	ncdim fdims;		/* name and size of dimension */
    ncatt att;			/* attribute */
    ncatt att0;			/* initialize attribute */
    ncvar var;			/* variable */
	int bad_file_attr = 243; /* matches merr_badfileatt in tmap_errors.parm*/
	
	strcpy(nc.fername, name);
    strcpy(nc.fullpath, path);
    nc.fer_dsetnum = *setnum;

	/* Set attribute with initialization values*/

	strcpy(att0.name, " ");
	att0.type = NC_CHAR;
	att0.outtype = NC_CHAR;
	att0.attid = 0;
	att0.outflag = 0;
	att0.len = 1;
    att0.string = (char *) malloc((att0.len+1)* sizeof(char*));
    strcpy (att0.string," ");
    att0.vals = (double *) malloc(1 * sizeof(double));
    att0.vals[0] = 0;

    /*
     * get number of dimensions, number of variables, number of global
     * atts, and dimension id of unlimited dimension, if any
     */
    nc_status = nc_inq(*ncid, &nc.ndims, &nc.nvars, &nc.ngatts, &nc.recdim) ;
    if (nc_status != NC_NOERR) return nc_status;

    /* get dimension info */
    if (nc.ndims > 0) {
       for (i = 0; i < nc.ndims; i++) {
	   nc_status = nc_inq_dim(*ncid, i, fdims.name, &fdims.size); 
       if (nc_status != NC_NOERR) return nc_status;

	   if (nc_status != NC_NOERR) return nc_status;
	   strcpy (nc.dims[i].name, fdims.name);
	   nc.dims[i].size = fdims.size;
       }
	}

     nc.vars_list_initialized = FALSE;
	 nc_status = NC_NOERR;

   /* get info on global attributes, treat as pseudo-variable . list of attributes*/


    /* get global attributes */

	if (nc.ngatts > 0)
	{
       strcpy(var.name, ".");

       var.attrs_list_initialized = FALSE;

       var.type = NC_CHAR;
       var.outtype = NC_CHAR;
       var.varid = 0;
	   var.natts = nc.ngatts;
	   var.ndims = 1;
	   var.dims[0] = 1;
       var.has_fillval = FALSE;
       var.fillval = NC_FILL_FLOAT;
	   var.all_outflag = 1;
	   var.is_axis = FALSE;
	   var.axis_dir = 0;

	   var.attrs_list_initialized = FALSE;
       for (i = 0; i < nc.ngatts; i++)
          {

    /* initialize 
	      att = att0;*/

          nc_status = nc_inq_attname(*ncid, NC_GLOBAL, i, att.name);            
/*		  if (nc_status != NC_NOERR) fprintf(stderr, " ***NOTE: error reading global attribute id %d from file %s\n", i, nc.fullpath);  */
          if (nc_status == NC_NOERR) {

            att.attid = i+1;
            nc_status = nc_inq_att(*ncid, NC_GLOBAL, att.name, &att.type, &att.len);
/*            if (nc_status != NC_NOERR) fprintf(stderr, " ***NOTE: error reading global attribute %s from file %s\n",att.name, nc.fullpath); */
            if (nc_status == NC_NOERR) {

    /* Set output flag. By default only the global history attribute is written.
     *  For string attributes, allocate one more than the att.len, 
     *  presumably for the null terminator for the string (?) */

   	 	      att.outflag = 0;
		      if (strcmp(att.name,"history")==0)
		      {att.outflag = 1;
		      }

              if (att.len == 0) {	/* show 0-length attributes as empty strings */
  	          att.type = NC_CHAR;
	          att.outtype = NC_CHAR;
	          att.len = 1;
	          att.string = (char *) malloc(1* sizeof(char*));
              strcpy (att.string," ");
              }
              switch (att.type) {
              case NC_CHAR:
	          att.string = (char *) malloc((att.len+1)* sizeof(char*));
  
              nc_status = nc_get_att_text(*ncid, NC_GLOBAL, att.name, att.string );
              if (nc_status != NC_NOERR) return nc_status;
 
              break;
              default:
              att.vals = (double *) malloc(att.len * sizeof(double));

              nc_status = nc_get_att_double(*ncid, NC_GLOBAL, att.name, att.vals );
              if (nc_status != NC_NOERR) return nc_status;
 
              break;
              }

              }  /* end of the  if (nc_status == NC_NOERR) */
              }
      /*Save attribute in linked list of attributes for variable . (global attributes)*/	
           if (!var.attrs_list_initialized) {
              if ( (var.varattlist = list_init()) == NULL ) {
                fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize GLOBAL attributes list.\n");
                return_val = -1;
                return return_val; 
              }
            var.attrs_list_initialized = TRUE;
  	      }
   
           list_insert_after(var.varattlist, &att, sizeof(ncatt));
       }    /* global attributes list complete */

      /*Save variable in linked list of variables for this dataset */	
       if (!nc.vars_list_initialized) {
          if ( (nc.dsetvarlist = list_init()) == NULL ) {
            fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize variable list.\n");
            return_val = -1;
            return return_val; 
          }
          nc.vars_list_initialized = TRUE;
        }

       list_insert_after(nc.dsetvarlist, &var, sizeof(ncvar));

     }    

    /* get info on variables */
	
	if (nc.nvars > 0)
       for (iv = 0; iv < nc.nvars; iv++)
          {
		  nc_status = nc_inq_var(*ncid, iv, var.name, &var.type, &var.ndims,
			     var.dims, &var.natts);
          if (nc_status != NC_NOERR) return nc_status;

          var.varid = iv+1;  
		  var.outtype = NC_FLOAT;
		  if (var.type == NC_CHAR) var.outtype = NC_CHAR;

		  /* is this a coordinate variable? 
		  */
          if (nc.ndims > 0) {
			 var.is_axis = FALSE;
	         var.axis_dir = 0;
			 i = 0;
             while (i < nc.ndims && var.is_axis == FALSE) {
	            if  (strcasecmp(var.name, nc.dims[i].name) == 0) var.is_axis = TRUE;
				i = i + 1;
			 }
		  }

		  /* get _FillValue attribute */

		  nc_status =  nc_inq_att(*ncid,iv,"_FillValue",&att.type,&att.len);

		  if(nc_status == NC_NOERR &&
		   att.type == var.type && att.len == 1) {
			  
			att.outflag = 1;
			att.outtype = NC_FLOAT;
		    var.has_fillval = TRUE;
			if(var.type == NC_CHAR) {
				att.outtype = NC_CHAR;
				nc_status = nc_get_att_text(*ncid, iv, "_FillValue",
						  &fillc );
				if (nc_status != NC_NOERR)  /* on error set attr to empty string */
				{ att.type = NC_CHAR;
				  att.outtype = NC_CHAR;
				  att.len = 1;
				  att.string = (char *) malloc((att.len+1)* sizeof(char*));
				  strcpy (att.string," ");
                  att.vals = (double *) malloc(1 * sizeof(double));
				  att.vals[0] = 0;
				  return_val = bad_file_attr;
				}
		    } else {
				nc_status = nc_get_att_double(*ncid, iv, "_FillValue",
						    &var.fillval ); }
				att.string = (char *) malloc(2*sizeof(char*));
				strcpy(att.string," ");
		    }
		  else  /* set to default NC value*/ 
			  {
			 var.has_fillval = FALSE;
		     switch (var.type) {
		     case NC_BYTE:
			/* don't do default fill-values for bytes, too risky */
			    var.has_fillval = 0;
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
		     default:
			 break;
			 att.string = (char *) malloc(2*sizeof(char*));
			 strcpy (att.string, " ");
		     }
		  }
 
		  var.all_outflag = 1;

          /* get all variable attributes 
           *  For string attributes, allocate one more than the att.len, 
           *  presumably for the null terminator for the string (?)
		   */
		  var.attrs_list_initialized = FALSE;

          if (var.natts > 0)
          {
          for (ia = 0; ia < var.natts; ia++)

          {

    /* initialize
	      att = att0; */

          nc_status = nc_inq_attname(*ncid, iv, ia, att.name);
/*		  if (nc_status != NC_NOERR) fprintf(stderr, " ***NOTE: error reading attribute id %d for variable %s, file %s\n", ia, var.name, nc.fullpath); */
          if (nc_status == NC_NOERR) {
		    att.attid = ia+1;

            nc_status = nc_inq_att(*ncid, iv, att.name, &att.type, &att.len);
/*            if (nc_status != NC_NOERR) fprintf(stderr, " ***NOTE: error reading attribute %s for variable %s, file %s\n",att.name, var.name, nc.fullpath); */
            if (nc_status == NC_NOERR) {

              if (att.len == 0) {	/* set 0-length attributes to empty strings */
	          att.type = NC_CHAR;
	          att.outtype = NC_CHAR;
	          att.len = 1;
	          att.string = (char *) malloc(1* sizeof(char*));
              strcpy (att.string," ");
              }
              switch (att.type) {
              case NC_CHAR:
	          att.string = (char *) malloc((att.len+1)* sizeof(char*));

              nc_status = nc_get_att_text(*ncid, iv, att.name, att.string );
              if (nc_status != NC_NOERR) /* on error set attr to empty string*/
			      {att.type = NC_CHAR;
			       att.outtype = NC_CHAR;
			       att.len = 1;
			       att.string = (char *) malloc((att.len+1)* sizeof(char*));
			       strcpy (att.string, " ");
			       return_val = bad_file_attr;
			       }
			   
              att.vals = (double *) malloc(1 * sizeof(double));
              att.vals[0] = 0;

              break;
              default:
		      att.outtype = NC_FLOAT;
              att.vals = (double *) malloc(att.len * sizeof(double));

              nc_status = nc_get_att_double(*ncid, iv, att.name, att.vals );
              if (nc_status != NC_NOERR) /* on error set attr to empty string*/
	             {att.type = NC_CHAR;
	             att.outtype = NC_CHAR;
	             att.len = 1;
	             att.string = (char *) malloc((att.len+1)* sizeof(char*));
	             strcpy (att.string, " ");
	             return_val = bad_file_attr;
	             }
	          att.string = (char *) malloc(2*sizeof(char*));
		      strcpy(att.string, " ");
              break;
              }

    /* Initialize output flag. Attributes written by default by Ferret
	   will be set to outflag = 1. 
	*/
              att.outflag = initialize_output_flag (att.name);

              } /* end of the if (nc_status == NC_NOERR)  */
              }
      /*Save attribute in linked list of attributes for this variable */	
           if (!var.attrs_list_initialized) {
              if ( (var.varattlist = list_init()) == NULL ) {
                fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize variable attributes list.\n");
                return_val = -1;
                return return_val; 
              }
              var.attrs_list_initialized = TRUE;
            }

           list_insert_after(var.varattlist, &att, sizeof(ncatt));
       }    /* variable attributes list complete */
     }  /* if var.natts > 0*/ 

      /*Save variable in linked list of variables for this dataset */	
       if (!nc.vars_list_initialized) {
          if ( (nc.dsetvarlist = list_init()) == NULL ) {
            fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize variable list.\n");
            return_val = -1;
            return return_val; 
          }
          nc.vars_list_initialized = TRUE;
        }

       list_insert_after(nc.dsetvarlist, &var, sizeof(ncvar));

     }    /* variables list complete */

/* Add dataset to global nc dataset linked list*/ 
  if (!list_initialized) {
    if ( (GLOBAL_ncdsetList = list_init()) == NULL ) {
      fprintf(stderr, "ERROR: ncf_add_dset: Unable to initialize GLOBAL_ncDsetList.\n");
      return_val = -1;
      return return_val; 
	}
    list_initialized = TRUE;
  }

  list_insert_after(GLOBAL_ncdsetList, &nc, sizeof(ncdset));
  return return_val;
  }

/* ----
 * Initialize new dataset to contain a non-netcdf dataset 
 * save in GLOBAL_ncdsetList for attribute handling 
 */

int FORTRAN(ncf_init_other_dset)(int *setnum, char name[], char path[])

{
  ncdset nc; 
  static int return_val=FERR_OK; /* static because it needs to exist after the return statement */
  
    int i;				/* loop controls */
	int ia;
	int iv;
    int nc_status;		/* return from netcdf calls */
    ncatt att;			/* attribute */
    ncvar var;			/* variable */

    strcpy(nc.fername, name);
    strcpy(nc.fullpath, path);
    nc.fer_dsetnum = *setnum;

    nc.ngatts = 1;
    nc.nvars = 0;
	nc.recdim = -1;   /* not used, but initialize anyway*/
	nc.ndims = 4;   
    nc.vars_list_initialized = FALSE;

   /* set one global attribute, treat as pseudo-variable . the list of variables */

       strcpy(var.name, ".");

       var.attrs_list_initialized = FALSE;

       var.type = NC_CHAR;
       var.outtype = NC_CHAR;
       var.varid = 0;
	   var.natts = nc.ngatts;
       var.has_fillval = FALSE;
       var.fillval = NC_FILL_FLOAT;
	   var.all_outflag = 1;
	   var.is_axis = FALSE;
	   var.axis_dir = 0;

	   var.attrs_list_initialized = FALSE; 

		  att.outflag = 1;
          att.type = NC_CHAR;
          att.outtype = NC_CHAR;
		  att.len = strlen(name);
          strcpy(att.name, name );

	      att.string = (char *) malloc((att.len+1)* sizeof(char*));
		  strcpy(att.string, name );

      /*Save attribute in linked list of attributes for variable .*/	
       if (!var.attrs_list_initialized) {
          if ( (var.varattlist = list_init()) == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize GLOBAL attributes list.\n");
            return_val = -1;
            return return_val; 
          }
          var.attrs_list_initialized = TRUE;
	  }

       list_insert_after(var.varattlist, &att, sizeof(ncatt));

       /* global attributes list complete */

      /*Save variable in linked list of variables for this dataset */	
       if (!nc.vars_list_initialized) {
          if ( (nc.dsetvarlist = list_init()) == NULL ) {
            fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize variable list.\n");
            return_val = -1;
            return return_val; 
          }
          nc.vars_list_initialized = TRUE;
        }

       list_insert_after(nc.dsetvarlist, &var, sizeof(ncvar));

/* Add dataset to global nc dataset linked list*/ 
  if (!list_initialized) {
    if ( (GLOBAL_ncdsetList = list_init()) == NULL ) {
      fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize GLOBAL_ncDsetList.\n");
      return_val = -1;
      return return_val; 
	}
    list_initialized = TRUE;
  }

  list_insert_after(GLOBAL_ncdsetList, &nc, sizeof(ncdset));
  return_val = FERR_OK;
  return return_val;
  }

/* ----
 * Find a dataset based on an integer id and return the nc_ptr.
 */
ncdset *ncf_ptr_from_dset(int *dset)
{
  static ncdset *nc_ptr=NULL;
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ncdsetList, dset, NCF_ListTraverse_FoundDsetID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    /* fprintf(stderr, "\nERROR: in ncf_ptr_from_dset: No dataset of id %d was found.\n\n", *dset); */
    return NULL;
  }

  nc_ptr=(ncdset *)list_curr(GLOBAL_ncdsetList); 
  
  return nc_ptr;
}

/* ----
 * Find a dataset based on its integer ID and return a pointer to its variable list
 */

LIST *ncf_get_ds_varlist( int *dset)
{
  ncdset *nc_ptr=NULL;
  static LIST *list_ptr=NULL;

  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL ) { return; }

  list_ptr=nc_ptr->dsetvarlist; 
  return list_ptr;
}

/* ----
 * Find a variable based on its dataset and variable IDs 
 * and return a pointer to its attribute list
 */

LIST *ncf_get_ds_var_attlist( int *dset, int *varid)
{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  static LIST *varlist=NULL;
  static LIST *att_ptr=NULL;
  int status;

   /*
   * Get the list of variables.  
   */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 

  att_ptr=var_ptr->varattlist; 
  return att_ptr;
}

/* ----
 * Remove a dataset from the global dataset list
 */

int FORTRAN(ncf_delete_dset)(int *dset)
{

  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  ncatt *att_ptr=NULL;
  static int return_val;
  LIST *varlist;
  LIST *dummy;
  int ivar;

/* Find the dataset
 */ 

  return_val = ATOM_NOT_FOUND;
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL ) { return return_val; }

  /* For each variable, deallocate the list of attributes, 
   * and remove the variable from the dataset list
   */

  varlist = ncf_get_ds_varlist(dset);
  var_ptr=(ncvar *)list_front(varlist); 
  for (ivar = 0; ivar< nc_ptr->nvars ;ivar ++ )
  {  
/*	 list_free(var_ptr->varattlist, LIST_DEALLOC); */ /* removed here just for testing...*/
     list_remove_curr(varlist);

     /* Point to next variable */
     dummy = list_mvnext(varlist);
     var_ptr=(ncvar *)list_curr(varlist); 
  }

/* Remove dataset from dataset list */


  list_remove_curr(GLOBAL_ncdsetList);

  return_val = FERR_OK;
  return return_val;
  }

/* ----
 * Add a new variable to the pseudo (user-variable) dataset.
 */
int  FORTRAN(ncf_add_var)( int *dset, int *varid, int *type, char varname[], char title[], char units[], double *bad)

{
  ncdset *nc_ptr=NULL;
  LIST *test_ptr=NULL;
  ncatt att;
  ncvar var;
  int status=LIST_OK;
  static int return_val;
  int *i;
  int newvar;
  int my_len;
  LIST *vlist=NULL;

   /*
   * Get the dataset pointer.  
   */
  return_val = ATOM_NOT_FOUND;  
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

   /*
   * Get the list of variables.  See if this variable already exists.
   */
  newvar = FALSE;
  vlist = ncf_get_ds_varlist(dset);
  status = list_traverse(vlist, varname, NCF_ListTraverse_FoundVarName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    newvar = TRUE;
  }

  if (newvar == TRUE)
  {
  nc_ptr->nvars = nc_ptr->nvars + 1;
  }
  else
   /* If this variable is not new, remove the old definition of it
   */
  {

  list_remove_curr(vlist);

  }

   /*
    * Set variable structure and insert the new variable at the end of the 
	* variable list. The type is not known at this time.
    */

  strcpy(var.name,varname);
  var.type = *type;
  var.outtype = *type;
  var.ndims = 4;
  var.natts = 0;  
  var.varid = *varid;
  var.is_axis = FALSE;   /* coordinate variable */
  var.axis_dir = 0;
  var.has_fillval = FALSE;
  var.all_outflag = 1;
  var.fillval = 0;  /* initialize this */
  var.attrs_list_initialized = FALSE;

  test_ptr = list_init(); 
  if ( (var.varattlist = list_init()) == NULL ) {
      fprintf(stderr, "ERROR: ncf_init_uvar_dset: Unable to initialize attributes list.\n");
      return_val = -1;
      return return_val; 
      }
  var.attrs_list_initialized = TRUE;

   /* Set up initial set of attributes*/

/*  Save the long_name, all variables
 *  For string attributes, allocate one more than the att.len, 
 *  presumably for the null terminator for the string (?) */

    var.natts = var.natts+1;

    strcpy(att.name, "long_name");    
	att.type = NC_CHAR; 
	att.outtype = NC_CHAR;
    att.attid = var.natts;
    att.outflag = 1;
    att.len = strlen(title);
    att.string = (char *) malloc((att.len+1)* sizeof(char*));
    strcpy(att.string, title);
	
    att.vals = (double *) malloc(1 * sizeof(double));
    att.vals[0] = 0; 

    /*Save attribute in linked list of attributes for variable .*/	

    list_insert_after(var.varattlist, &att, sizeof(ncatt));

/*  Now the units, if given
 *  For the units string, allocate one more than the att.len, 
 *  presumably for the null terminator for the string (?)*/

    if (strlen(units) > 0 )
		{
		var.natts = var.natts+1;

		att.attid = var.natts;
		strcpy(att.name, "units");
		att.len = strlen(units);
		att.outflag = 1;
		att.type = NC_CHAR;
		att.outtype = NC_CHAR;
		att.string = (char *) malloc((att.len+1)* sizeof(char*));
		strcpy(att.string, units);

        my_len = 1;
	    att.vals = (double *) malloc(my_len * sizeof(double)); 
        att.vals[0] = 0;


/*Save attribute in linked list of attributes for this variable */	

		list_insert_after(var.varattlist, &att, sizeof(ncatt));
		}


/* Now the missing_value, for numeric variables*/
/*    if (*type != NC_CHAR)
    { */
		var.natts = var.natts+1;
        var.fillval = *bad;

		att.attid = var.natts;
		strcpy(att.name,"missing_value");
		att.len = 1;
		att.type = NC_FLOAT;
		att.outtype = NC_FLOAT;
		att.vals = (double *) malloc(att.len * sizeof(double));
		att.vals[0] = *bad;

    /* Initialize output flag. Attributes written by default by Ferret
	   will be set to outflag = 1. 
	*/
          att.outflag = initialize_output_flag (att.name);

      /*Save attribute in linked list of attributes for this variable */	
		if (!var.attrs_list_initialized) {
		  if ( (var.varattlist = list_init()) == NULL ) {
            fprintf(stderr, "ERROR: add_var: Unable to initialize variable attributes list.\n");
            return_val = -1;
            return return_val; 
          }
          var.attrs_list_initialized = TRUE;
        }

       list_insert_after(var.varattlist, &att, sizeof(ncatt));
 /*   } */

/*Save variable in linked list of variables for this dataset */

    list_insert_after(nc_ptr->dsetvarlist, &var, sizeof(ncvar));
  
  return_val = FERR_OK;
  return return_val;
}

/* ----
 * Find a variable based on its variable ID and dataset ID
 * Add a new numeric attribute.
 */
int  FORTRAN(ncf_add_var_num_att)( int *dset, int *varid, char attname[], int *attype, int *attlen, int *outflag, float *vals)

{
  ncatt *att_ptr=NULL;
  ncvar *var_ptr=NULL;
  ncatt att;
  int status=LIST_OK;
  static int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;

   /*
    * Get the list of variables, find pointer to variable varid.
    */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  if (var_ptr->natts < 1) return ATOM_NOT_FOUND;

    /*
    * Get the list of attributes for the variable in the dataset
    * If the attribute is already defined, return -1* attid
    */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status == LIST_OK ) {\
    att_ptr=(ncatt *)list_curr(varattlist); 
    return_val = -1* att_ptr->attid; 
    return return_val;
    }

   /* Increment number of attributes.  
   */

  var_ptr->natts = var_ptr->natts + 1;

   /*
    * Set attribute structure and insert the new attribute at 
	* the end of the attribute list. 
    */

  strcpy(att.name,attname);
  att.attid = var_ptr->natts;
  att.type = *attype;
  att.outtype = NC_FLOAT;
  att.len = *attlen;
  att.outflag = *outflag;
  att.vals = (double *) malloc(*attlen * sizeof(double));

  for (i = 0; i<*attlen;i++ )
  {att.vals[i] = vals[i];
  }

   /*Save attribute in linked list of attributes for this variable */	

  list_insert_after(var_ptr->varattlist, &att, sizeof(ncatt));

  return_val = FERR_OK;
  return return_val;
}

/* ----
 * Find a variable  based on its variable ID and dataset ID
 * Add a new string attribute.
 */
int  FORTRAN(ncf_add_var_str_att)( int *dset, int *varid, char attname[], int *attype, int *attlen, int *outflag, char attstring[])

{
  ncatt *att_ptr=NULL;
  ncvar *var_ptr=NULL;
  ncatt att;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;

   /*
    * Get the list of variables, find pointer to variable varid.
    */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  
    /*
    * here if natts < 1 we should initialize the list!
    */
      /*Save attribute in linked list of attributes for variable */	
  if (!var_ptr->attrs_list_initialized) {
    if ( (var_ptr->varattlist = list_init()) == NULL ) {
      fprintf(stderr, "ERROR: add_var_str_att: Unable to initialize attributes list.\n");
      return_val = -1;
      return return_val; 
     }
    var_ptr->attrs_list_initialized = TRUE;
  }

    /*
    * Get the list of attributes for the variable in the dataset
    * If the attribute is already defined, return -1* attid
    */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status == LIST_OK ) {\
    att_ptr=(ncatt *)list_curr(varattlist); 
    return_val = -1* att_ptr->attid; 

    return return_val;
    }

   /* Increment number of attributes.  
   */

  var_ptr->natts = var_ptr->natts + 1;

   /*
    * Set attribute structure and insert the new attribute at 
	* the end of the attribute list. 
	
    *  For string attributes, allocate one more than the att.len, 
    *  presumably for the null terminator for the string (?)
    */

  strcpy(att.name,attname);
  att.attid = var_ptr->natts;
  att.type = *attype;
  att.outtype = NC_FLOAT;
  att.len = *attlen;
  att.outflag = *outflag;
  att.string = (char *) malloc((att.len+1)* sizeof(char*));
  strcpy(att.string, attstring);

      /*Save attribute in linked list of attributes for this variable */	


 list_insert_after(var_ptr->varattlist, &att, sizeof(ncatt));

  return_val = FERR_OK;
  return return_val;
}


/* ----
 * Find an attribute based on its variable ID and dataset ID
 * Replace the type, length, and/or value(s).
 */
int  FORTRAN(ncf_repl_var_att)( int *dset, int *varid, char attname[], int *attype, int *attlen, float *vals, char attstring[])

{
  ncatt *att_ptr=NULL;
  ncvar *var_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;

   /*
    * Get the list of variables, find pointer to variable varid.
    */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  if (var_ptr->natts < 1) return ATOM_NOT_FOUND;

   /*
    * Get the list of attributes for the variable in the dataset
    * If the attribute is not defined, return
    */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
    }

   /*
    * Get the attribute.
    */
  att_ptr=(ncatt *)list_curr(varattlist); 

   /*
    * Free the memory used by the string or values 
    */
  if (att_ptr->type == NC_CHAR)
  {
	  free(att_ptr->string);
  }
  else
  {
	  free(att_ptr->vals);
  }
  

   /*
    * Keep the name and ID. Reset type, length, and values
    *  For string attributes, allocate one more than the att.len, 
    *  presumably for the null terminator for the string (?)
    */

  att_ptr->type = *attype;
  att_ptr->outtype = NC_FLOAT;
  att_ptr->len = *attlen;

  if (*attlen == 0) /* set 0-length attributes to empty strings */
	  {
		  att_ptr->type = NC_CHAR;
		  att_ptr->outtype = NC_CHAR;
		  att_ptr->len = 1;
		  strcpy(att_ptr->string," ");
	  }
   else
	  {
	   switch (*attype) 
		   {
		   case NC_CHAR:
			   i = (*attlen+1);   /* this line for debugging*/
	        att_ptr->string = (char *) malloc((*attlen+1)* sizeof(char*));
            strcpy(att_ptr->string,attstring);
            break;
			
		   default:
            att_ptr->vals = (double *) malloc(*attlen * sizeof(double));
	        for (i = 0; i<*attlen;i++ )
            {
				att_ptr->vals[i] = vals[i];
            }
            break;
         }
	  }

  return_val = FERR_OK;
  return return_val;
}

/* ---- 
 * Find an attribute based on its variable ID and dataset ID
 * Delete it.
 */
int  FORTRAN(ncf_delete_var_att)( int *dset, int *varid, char attname[])

{
  ncatt *att_ptr=NULL;
  ncvar *var_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;

   /*
    * Get the list of variables, find pointer to variable varid.
    */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  if (var_ptr->natts < 1) return ATOM_NOT_FOUND;

   /*
    * Get the list of attributes for the variable in the dataset
    * If the attribute is not defined, return
    */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
    }

   /*
    * Get the attribute.
    */

/* Remove it
 */
  list_remove_curr(varattlist);

  
   /* Decrement number of attributes.  
   */

  var_ptr->natts = var_ptr->natts - 1;


  return_val = FERR_OK;
  return return_val;
  }


/* ---- 
 * Find an attribute based on its variable ID and dataset ID
 * Change its output flag: 1=output it, 0=dont.
 */
int  FORTRAN(ncf_set_att_flag)( int *dset, int *varid, char attname[], int *attoutflag)

{
  ncatt *att_ptr=NULL;
  ncvar *var_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist;
  LIST *varattlist;

   /*
    * Get the list of variables, find pointer to variable varid.
    */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 
  if (var_ptr->natts < 1) return ATOM_NOT_FOUND;

   /*
    * Get the list of attributes for the variable in the dataset
    * If the attribute is not defined, return
    */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  status = list_traverse(varattlist, attname, NCF_ListTraverse_FoundVarAttName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
    }

   /*
    * Get the attribute.
    */
  att_ptr=(ncatt *)list_curr(varattlist); 

   /*
    * Keep the attribute as is, but reset its output flag.
    */

  att_ptr->outflag = *attoutflag;


  return_val = FERR_OK;
  return return_val;
}



/* ---- 
 * Find variable based on its variable ID and dataset ID
 * Change the variable flag: 
 * 1=output no attributes, 
   0=check individual attribute output flags,
   2=write all attributes
*  3=reset attr flags to defaults
 */
int  FORTRAN(ncf_set_var_out_flag)( int *dset, int *varid, int *all_outflag)

{
  ncvar *var_ptr=NULL;
  ncatt *att_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  int *iatt;
  LIST *varlist;
  LIST *varattlist;

   /*
    * Get the list of variables, find pointer to variable varid.
    */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 

   /*
    * Keep the default if there are no attributes
    */

  if (var_ptr->natts < 1)
  {
	  var_ptr->all_outflag = 1;
	  return FERR_OK;
  }

   /*
    * Reset the variable output flag.
    */
  var_ptr->all_outflag = *all_outflag;
  if (*all_outflag == 0)

  {
   /*
    * Get the list of attributes for the variable varid
    */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  /* 
   * reset the output flag for each attribute
   */
  for (i = 1; i <= var_ptr->natts; i++ )
	  {
	  /* *iatt = i; */

	  status = list_traverse(varattlist, &i, NCF_ListTraverse_FoundVarAttID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
	  if ( status != LIST_OK ) 
		  {
		  return_val = ATOM_NOT_FOUND;
		  return return_val;
		  }

   /*
    * Get the attribute.
    */
	  att_ptr=(ncatt *)list_curr(varattlist); 

	  /*
	  * Reset the attribute output flag.
	  */

	  att_ptr->outflag = 0;

	  }  /* end of iatt loop*/   

  }


  else if (*all_outflag == 2)

  {
   /*
    * Get the list of attributes for the variable varid
    */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  /* 
   * reset the output flag for each attribute
   */

  for (i = 1; i <= var_ptr->natts; i++ )
 {
	 /* *iatt = i; */
	 status = list_traverse(varattlist, &i, NCF_ListTraverse_FoundVarAttID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
	  if ( status != LIST_OK ) 
		  {
		  return_val = ATOM_NOT_FOUND;
		  return return_val;
		  }

   /*
    * Get the attribute.
    */

	  att_ptr=(ncatt *)list_curr(varattlist); 

	  /*
	  * Reset the attribute output flag.
	  */

	  att_ptr->outflag = 1;

	  }  /* end of iatt loop*/   

 }

  else if  (*all_outflag == 3)

  {

   /*
    * Get the list of attributes for the variable varid
    */
  varattlist = ncf_get_ds_var_attlist(dset, varid);

  /* 
   * reset the output flag for each attribute to the default Ferret value 
   */
  for (i = 1; i <= var_ptr->natts; i++ )
	  {
	  /* *iatt = i; */
	  status = list_traverse(varattlist, &i, NCF_ListTraverse_FoundVarAttID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
	  if ( status != LIST_OK ) {
	    return_val = ATOM_NOT_FOUND;
	    return return_val;
	  }

   /*
    * Get the attribute.
    */
	  att_ptr=(ncatt *)list_curr(varattlist); 

	  /*
	  * Reset the attribute output flag to the Ferret default value
	    (output missing flag, etc, but not nonstd attributes from
		the intput file or user definitions.)
	  */

	  att_ptr->outflag = initialize_output_flag(att_ptr->name);

	  }  /* end of iatt loop*/   
  }

  return_val = FERR_OK;
  return return_val;
}


/* ---- 
 * Find variable based on its variable ID and dataset ID
 * Change the variable output type.
 */
int  FORTRAN(ncf_set_var_outtype)( int *dset, int *varid, int *outtype)

{
  ncvar *var_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;

   /*
    * Get the list of variables, find pointer to variable varid.
    */
  varlist = ncf_get_ds_varlist(dset);

  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 

   /*
    * Reset the variable output type.
    */
  var_ptr->outtype = *outtype;

  return_val = FERR_OK;
  return return_val;
}

/* ---- 
 * Find variable based on its variable ID and dataset ID
 * Check that its a coordinate variable and set the axis direction.
 */

int  FORTRAN(ncf_set_axdir)( int *dset, int *varid, int *axdir)

{
  ncvar *var_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;

   /*
    * Get the list of variables, find pointer to variable varid.
    */
  varlist = ncf_get_ds_varlist(dset);

  return_val = ATOM_NOT_FOUND;
  status = list_traverse(varlist, varid, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr=(ncvar *)list_curr(varlist); 

   /*
    * Reset the variable output type.
    */
  return_val = ATOM_NOT_FOUND;
  if (var_ptr->is_axis)
  {
	 var_ptr->axis_dir = *axdir;
	 return_val = FERR_OK;
  }

  return return_val;
}

/* ---- 
 * Find an attribute based on its dataset ID, variable ID and attribute ID
 * Add the attribute to variable 2 in dataset 2
 */
int  FORTRAN(ncf_transfer_att)(int *dset1, int *varid1, int *iatt, int *dset2, int *varid2)

{
  ncatt *att_ptr1=NULL;
  ncatt att;
  ncvar *var_ptr1=NULL;
  ncvar *var_ptr2=NULL;
  int status=LIST_OK;
  int return_val;
  int i;
  LIST *varlist1;
  LIST *varlist2;
  LIST *varattlist1;
  LIST *varattlist2;

   /*
    * Get the list of variables in dset1, find pointer to variable varid1.
    */
  varlist1 = ncf_get_ds_varlist(dset1);

  status = list_traverse(varlist1, varid1, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr1=(ncvar *)list_curr(varlist1); 
  if (var_ptr1->natts < 1) return ATOM_NOT_FOUND;

   /*
    * Get the list of attributes for the variable varid1
    * If the attribute is not defined, return
    */
  varattlist1 = ncf_get_ds_var_attlist(dset1, varid1);

  status = list_traverse(varattlist1, iatt, NCF_ListTraverse_FoundVarAttID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
    }

   /*
    * Get the attribute.
    */
  att_ptr1=(ncatt *)list_curr(varattlist1); 

   /*
    * Get the list of variables in dset2, find pointer to variable varid2
    */
  varlist2 = ncf_get_ds_varlist(dset2);

  status = list_traverse(varlist2, varid2, NCF_ListTraverse_FoundVarID, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) return;

  var_ptr2=(ncvar *)list_curr(varlist2); 
  
   /*
    * Get the list of attributes for the variable varid2
    */
  varattlist2 = ncf_get_ds_var_attlist(dset2, varid2);

   /* Increment number of attributes for varid2
   */

  var_ptr2->natts = var_ptr2->natts + 1;

   /*
    * Set attribute structure and insert the new attribute at 
	* the end of the attribute list. 
    *  For string attributes, allocate one more than the att.len, 
    *  presumably for the null terminator for the string (?)
    */
  strcpy(att.name, att_ptr1->name);
  att.attid = var_ptr2->natts;
  att.type = att_ptr1->type;
  att.outtype = att_ptr1->type;
  att.len = att_ptr1->len;
  att.outflag = att_ptr1->outflag;
  
  if (att_ptr1->type == NC_CHAR)
  {
	  att.string = (char *) malloc((att_ptr1->len+1)* sizeof(char*)); 
	  strcpy(att.string, att_ptr1->string);
  }
  else
  {
	  att.vals = (double *) malloc(att_ptr1->len * sizeof(double));
	  for (i = 0; i<att_ptr1->len;i++ )
		  {att.vals[i] = att_ptr1->vals[i];
		  }
  }

  /*Save attribute in linked list of attributes for this variable */	

  list_insert_after(var_ptr2->varattlist, &att, sizeof(ncatt));

  return_val = FERR_OK;
  return return_val;
}


/* ---- 
 * Find variable based on the dataset ID and variable name
 * Delete it.
 */
int  FORTRAN(ncf_delete_var)( int *dset, char varname[])

{
  ncdset *nc_ptr=NULL;
  ncvar *var_ptr=NULL;
  ncatt *att_ptr=NULL;
  int status=LIST_OK;
  int return_val;
  LIST *varlist;

 /* Find the dataset based on its integer ID 
 */

  return_val = ATOM_NOT_FOUND;  
  if ( (nc_ptr = ncf_ptr_from_dset(dset)) == NULL )return return_val;

   /*
   * Get the list of variables. Find varname in the dataset.
   */
  varlist = ncf_get_ds_varlist(dset);
  status = list_traverse(varlist, varname, NCF_ListTraverse_FoundVarName, (LIST_FRNT | LIST_FORW | LIST_ALTR));
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }
  
  /* Deallocate the list of attributes, and remove the 
   * variable from the dataset list
   */
  var_ptr=(ncvar *)list_curr(varlist); 
/*  list_free(var_ptr->varattlist, LIST_DEALLOC); */ /* removed just for debugging...*/

  list_remove_curr(varlist);
  
   /* Decrement number of variables in the dataset.  
    */

  nc_ptr->nvars = nc_ptr->nvars - 1;

  return_val = FERR_OK;
  return return_val;
  }


/* ---- 
 * For attributes that Ferret always writes, set the output flag to 1
   All others are not written by default. The flag can be set to 1 by the user.
   The modulo flag is set to 0. This will be overriden ni the Ferret code
   depending on the value of the modulo attribute.
  */

int initialize_output_flag (char *attname)
{
	int return_val;
    return_val = 0;

    /* attributes on coordinate variables */
	if (strcmp(attname,"axis")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"units")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"calendar")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"positive")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"point_spacing")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"modulo")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"time_origin")==0)
	{return_val = 1;
	}

    /* attributes on variables */
	if (strcmp(attname,"missing_value")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"_FillValue")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"long_name")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"title")==0)
	{return_val = 1;
	}
	if (strcmp(attname,"history")==0)
	{return_val = 1;
	}
	return return_val;

}


/* ---- 
 * See if the name in data matches the ferret dset name in 
 * curr. Ferret always capitalizes everything so be case INsensitive.
 */
int NCF_ListTraverse_FoundDsetName( char *data, char *curr )
{
  ncdset *nc_ptr=(ncdset *)curr; 

  if ( !strcasecmp(data, nc_ptr->fername) ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}


/* ---- 
 * See if the dataset id in data matches the ferret dset id in curr.
 */
int NCF_ListTraverse_FoundDsetID( char *data, char *curr )
{
  ncdset *nc_ptr=(ncdset *)curr; 
  int ID=*((int *)data);

  if ( ID == nc_ptr->fer_dsetnum ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}


/* ---- 
 * See if the name in data matches the variable name in 
 * curr. Ferret always capitalizes everything so be case INsensitive,
 * unless the string has been passed in inside single quotes.
 */
int NCF_ListTraverse_FoundVarName( char *data, char *curr )
{
  ncvar *var_ptr=(ncvar*)curr;

  if ( !strcasecmp(data, var_ptr->name) ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}

/* ---- 
 * See if the name in data matches the variable name in 
 * curr. Make the string comparison case-sensive.
 */
int NCF_ListTraverse_FoundVarNameCase( char *data, char *curr )
{
  ncvar *var_ptr=(ncvar*)curr;

  if ( !strcmp(data, var_ptr->name) ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}

/* ---- 
 * See if the ID in data matches the variable ID in curr. 
 */
int NCF_ListTraverse_FoundVarID( char *data, char *curr )
{
  ncvar *var_ptr=(ncvar*)curr; 
  int ID=*((int *)data);

   if ( ID == var_ptr->varid)  {
    return FALSE; /* found match */
  } else
    return TRUE;
}


/* ---- 
 * See if the name in data matches the attribute name in curr.
 */
int NCF_ListTraverse_FoundVarAttName( char *data, char *curr )
{
  ncatt *att_ptr=(ncatt *)curr;

  if ( !strcasecmp(data, att_ptr->name) ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}

/* ---- 
 * See if the name in data matches the attribute name in curr. 
 * Make the string comparison case-sensive.
 */
int NCF_ListTraverse_FoundVarAttNameCase( char *data, char *curr )
{
  ncatt *att_ptr=(ncatt *)curr;

  if ( !strcmp(data, att_ptr->name) ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}
/* ---- 
 * See if there is an ID in data matches the attribute id in curr.
 */
int NCF_ListTraverse_FoundVarAttID( char *data, char *curr )
{
  ncatt *att_ptr=(ncatt *)curr;
  int ID=*((int *)data);

  if ( ID== att_ptr->attid)  {
    return FALSE; /* found match */
  } else
    return TRUE;
}
