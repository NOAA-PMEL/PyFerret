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

/* NC_Util.h
 *
 * Ansley Manke
 * Ferret V600 April 2005
 *
 * This is the header file to be included by routines which
 * are part of the Ferret NetCDF attribute handling library.
 *
 */

/* .................... Defines ..................... */

#define TRUE  1
#define FALSE 0
#define YES   1
#define NO    0

#define LO    0
#define HI    1

#define ATOM_NOT_FOUND 0  /* This should match the atom_not_found parameter in ferret.parm. */
#define FERR_OK 3  /* This should match the ferr_ok parameter in errmsg.parm. */
#define DATSET -1  /* This should match the NC_GLOBAL parameter in netcdf.h */

#define MAX_PATH_NAME	2048	 /* max length of a path */
#define MAX_FER_SETNAME	256	 /* max length of a path */
#define MAX_FER_SETNAME	256	 /* max length of a path */


/* .................... Typedefs .................... */


/* dataset */
/*
typedef struct  {			
    char fername[MAX_FER_SETNAME];
	int fer_dsetnum;
} ncdset;
*/

typedef struct  {			/* dimension */
    char name[NC_MAX_NAME];
    size_t size;
} ncdim;

typedef struct  {
	char fullpath[MAX_PATH_NAME];
    char fername[MAX_FER_SETNAME];
	LIST *dsetvarlist;
    ncdim dims[NC_MAX_DIMS];
	int ndims;
    int ngatts;
	int recdim;
	int nvars;
	int vars_list_initialized;
	int fer_dsetnum;
	int fer_current;
	int its_epic;
} ncdset;

typedef struct  {          /* variable */
    char name[NC_MAX_NAME];
	LIST *varattlist;
	nc_type type;
	int outtype;
    int ndims;
    int dims[MAX_VAR_DIMS];
    int natts;
	int varid;
	int is_axis;           /* coordinate variable */
	int axis_dir;          /* coordinate direction 1,2,3,4 for X,Y,Z,T */
    int has_fillval;
	int all_outflag;       /* 0 write no attrs, 
	                          1 check individual attr flags
							  2 write all attrs,
	                          3 reset attr flags to Ferret defaults
                           */
    double fillval;
	int attrs_list_initialized;
} ncvar;

typedef struct {			/* attribute */
    char name[NC_MAX_NAME];
    nc_type type;
    int outtype;
	int attid;
	int outflag;        /* 1 to write this attr, 0 to not write */
    size_t len;
    char *string;       /* for text attributes (type = NC_CHAR) */
    double *vals;       /* for numeric attributes of all types */
} ncatt;


#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

