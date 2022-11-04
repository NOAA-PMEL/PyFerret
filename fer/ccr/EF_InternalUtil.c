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


/* EF_InternalUtil.c
 *
 * Jonathan Callahan
 * Sep 4th 1997
 *
 * This file contains all the utility functions which Ferret
 * needs in order to communicate with an external function.
 */

/* Ansley Manke  March 2000
 *  Additions to allow internally linked external functions.
 *  Source code is in the directory FERRET/fer/efi
 *  In that directory, run the perl script int_dlsym.pl
 *  int_dlsym.pl ./ > intlines.c
 *  The result is lines of C code to be put into this file.
 *  Search for the comment string --------------------
 *
 *  1.  Function declaration lines.  Need to edit these to have
 *      the correct number of arguments for the _compute subroutines.
 *  2.  definition of N_INTEF and structure I_EFnames
 *  3.  internal_dlsym lines at the end

* Jonathan Callahan and Ansley Manke  30-May-2000
 * Fix memory leak:  already_have_internals needs to be tested for when
 * we find the external function in efcn_gather_info  and set TRUE once
 * the internals have been set for the first time, also in efcn_gather_info.

* Ansley Manke  August 2001
 * add EOF_SPACE, EOF_STAT, EOF_TFUNC to the functions that are
 * statically linked

* V5.4 *acm* 10/01 add compress* to the statically linked fcns
* v6.0 *acm*  5/06 many more functions internally linked.
* V6.0 *acm*  5/06 string results for external functions
* v6.0 *acm*  5/06 internal_dlsym was missing the nco functions.
* V6.03 *acm& 5/07 Add tax_ functions, fill_xy to the statically-linked functions
* V6.07 *acm* 8/07 remove xunits_data from list of I_EFnames; it should never
*                  have been there.
* V6.12 *acm* 8/07 add functions scat2grid_bin_xy and scat2grid_nobs_xy.F
* V6.2 *acm* 11/08 New functions XCAT_STR, YCAT_STR, ...
* V6.2 *acm* 11/08 New internally-called function efcn_get_alt_type_fcn to
*                  get the name of a function to call if the arguments are of
*                  a different type than defined in the current function. E.g.
*                  this lets the user reference XCAT with string arguments and
*                  Ferret will run XCAT_STR
* V6.6 *acm* 4/10 add functions scat2grid_nbin_xy and scat2grid_nbin_xyt.F
* V664 *kms*  9/10 Added python-backed external functions via $FER_DIR/lib/libpyefcn.so
*                  Made external function language check more robust
*                  Check that GLOBAL_ExternalFunctionsList is not NULL in ef_ptr_from_id_ptr
*      *kms* 11/10 Check for libpyefcn.so in $FER_LIBS instead of $FER_DIR/lib
*      *kms* 12/10 Eliminated libpyefcn.so; link to pyefcn static library
*                  This makes libpython... a required library.
* *acm*  1/12      - Ferret 6.8 ifdef double_p for double-precision ferret, see the
*                  definition of macro DFTYPE in ferret.h
*      *kms*  3/12 Add E and F dimensions
*      *acm*  6/14 New separate function for DSG files
*      *acm*  9/14 Make DATE1900 accept an array of date strings, returning an array of coordinates
*      *acm*  2/15 TAX_DATESTRING works on an F or a T axis
*      *acm*  2/15 new Functions TIME_REFORMAT, FT_TO_ORTHOGONAL
* V702 *sh*   1/17 added support for FORTRAN90 dynamic memory management
*                  removing "memory" pointer in favor of individual arg ptrs
* V74  *acm*  2/18 New box_edges function
* V751 *acm*  7/19 remove ancient scat2gridgauss_*_V0 functions
* V751 *acm*  5/20 remove samplexy_closest function, renamed samplexy_nrst

* 10/2022 *acm* Code cleanup: defined length of cmd
*/


/* .................... Includes .................... */

#include <Python.h> /* make sure Python.h is first */
#include <sys/types.h>        /* required for some of our prototypes */
#include <sys/stat.h>
#include <sys/errno.h>

#include <ctype.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ferret.h"
#include "FerMem.h"
#include "EF_Util.h"
#include "list.h"		/* locally added list library */
#include "pyferret.h"		/* python external funtion interfaces */


/* ................ Global Variables ................ */
/*
 * The mr_list_ptr and cx_list_ptr are obtained from Ferret
 * and cached whenever they are passed into one of the "efcn_" functions.
 * These pointers can be accessed by the utility functions in efn_ext/.
 * This way the EF writer does not need to see these pointers.
 *
 * 1/17 - c argument pointers (GLOBAL_arg_ptrs) and the result pointer
 *        (GLOBAL_res_ptr)are obtained from FORTRAN later on demand
 *
 * This is the instantiation of these values.
 */

int    *GLOBAL_mr_list_ptr;
int    *GLOBAL_cx_list_ptr;
int    *GLOBAL_mres_ptr;
DFTYPE *GLOBAL_bad_flag_ptr;

DFTYPE *GLOBAL_arg_ptrs[EF_MAX_ARGS];
DFTYPE *GLOBAL_res_ptr;

static LIST *STATIC_ExternalFunctionList;

/*
 * The jumpbuffer is used by setjmp() and longjmp().
 * setjmp() is called by FORTRAN(efcn_compute)() in EF_InternalUtil.c and
 * saves the stack environment in jumpbuffer for later use by longjmp().
 * This allows one to bail out of external functions and still
 * return control to Ferret.
 * Check "Advanced Progrmming in the UNIX Environment" by Stevens
 * sections 7.10 and 10.14 to understand what's going on with these.
 */
static jmp_buf jumpbuffer;
static sigjmp_buf sigjumpbuffer;
static volatile sig_atomic_t canjump;

static int I_have_scanned_already = FALSE;

static void *internal_dlsym(char *name);

/* ............. Function Declarations .............. */
/*
 * Note that all routines called directly from Ferret,
 * ie. directly from Fortran, should be all lower case,
 * be of type 'void', pass by reference and should end with
 * an underscore.
 */


/* .... Functions called by Ferret .... */

int  FORTRAN(efcn_scan)( int * );
int  FORTRAN(efcn_already_have_internals)( int * );

void FORTRAN(create_pyefcn)(char fname[], int *lenfname, char pymod[], int *lenpymod,
                            char errstring[], int *lenerrstring);

int  FORTRAN(efcn_gather_info)( int * );
void FORTRAN(efcn_get_custom_axes)( int *, int *, int * );
void FORTRAN(efcn_get_result_limits)( int *, int *, int *, int * );
void FORTRAN(efcn_compute)( int *, int *, int *, int *, int *, DFTYPE *, int * );


void FORTRAN(efcn_get_custom_axis_sub)( int *, int *, double *, double *, double *, char *, int * );

int  FORTRAN(efcn_get_id)( char * );
int  FORTRAN(efcn_match_template)( int *, char * );

void FORTRAN(efcn_get_name)( int *, char * );
void FORTRAN(efcn_get_version)( int *, DFTYPE * );
void FORTRAN(efcn_get_descr)( int *, char * );
void FORTRAN(efcn_get_alt_type_fcn)( int *, char * );
int  FORTRAN(efcn_get_num_reqd_args)( int * );
void FORTRAN(efcn_get_has_vari_args)( int *, int * );
void FORTRAN(efcn_get_axis_will_be)( int *, int * );
void FORTRAN(efcn_get_axis_reduction)( int *, int * );
void FORTRAN(efcn_get_piecemeal_ok)( int *, int * );

void FORTRAN(efcn_get_axis_implied_from)( int *, int *, int * );
void FORTRAN(efcn_get_axis_extend_lo)( int *, int *, int * );
void FORTRAN(efcn_get_axis_extend_hi)( int *, int *, int * );
void FORTRAN(efcn_get_axis_limits)( int *, int *, int *, int * );
int  FORTRAN(efcn_get_arg_type)( int *, int *);
void FORTRAN(efcn_get_arg_name)( int *, int *, char * );
void FORTRAN(efcn_get_arg_unit)( int *, int *, char * );
void FORTRAN(efcn_get_arg_desc)( int *, int *, char * );
int  FORTRAN(efcn_get_rtn_type)( int *);
void FORTRAN(efcn_rqst_mr_ptrs)( int *, int *, int * ); // narg, mr_list, mres
void FORTRAN(efcn_pass_arg_ptr)(int *, DFTYPE *);
void FORTRAN(efcn_pass_res_ptr)(DFTYPE *);


/* .... Functions called internally .... */

/* Fortran routines from the efn/ directory */
void FORTRAN(efcn_copy_array_dims)(void);
void FORTRAN(efcn_set_work_array_dims)(int *, int *, int *, int *, int *, int *, int *,
                                              int *, int *, int *, int *, int *, int *);
void FORTRAN(efcn_get_workspace_addr)(DFTYPE *, int *, DFTYPE *);

static void EF_signal_handler(int signo);
static void (*fpe_handler)(int);      /* function pointers */
static void (*segv_handler)(int);
static void (*int_handler)(int);
static void (*bus_handler)(int);
int EF_Util_setsig();
int EF_Util_ressig();


void EF_store_globals(int *, int *, int *, DFTYPE *);

ExternalFunction *ef_ptr_from_id_ptr(int *);

int  EF_ListTraverse_fprintf( char *, char * );
int  EF_ListTraverse_FoundName( char *, char * );
int  EF_ListTraverse_MatchTemplate( char *, char * );
int  EF_ListTraverse_FoundID( char *, char * );

int  EF_New( ExternalFunction * );

/*  ------------------------------------
 *  Statically linked external functions
 *  Declarations generated by the perl script int_dlsym.pl.
 *  Need to fill out the arguments for the _compute subroutines.
 */
 
void FORTRAN(dsg_fmask_str_init)(int *);
void FORTRAN(dsg_fmask_str_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *);

void FORTRAN(ffta_init)(int *);
void FORTRAN(ffta_custom_axes)(int *);
void FORTRAN(ffta_result_limits)(int *);
void FORTRAN(ffta_work_size)(int *);
void FORTRAN(ffta_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(fftp_init)(int *);
void FORTRAN(fftp_custom_axes)(int *);
void FORTRAN(fftp_result_limits)(int *);
void FORTRAN(fftp_work_size)(int *);
void FORTRAN(fftp_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(fft_im_init)(int *);
void FORTRAN(fft_im_custom_axes)(int *);
void FORTRAN(fft_im_result_limits)(int *);
void FORTRAN(fft_im_work_size)(int *);
void FORTRAN(fft_im_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(fft_inverse_init)(int *);
void FORTRAN(fft_inverse_result_limits)(int *);
void FORTRAN(fft_inverse_work_size)(int *);
void FORTRAN(fft_inverse_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(fft_re_init)(int *);
void FORTRAN(fft_re_custom_axes)(int *);
void FORTRAN(fft_re_result_limits)(int *);
void FORTRAN(fft_re_work_size)(int *);
void FORTRAN(fft_re_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(sampleij_init)(int *);
void FORTRAN(sampleij_result_limits)(int *);
void FORTRAN(sampleij_work_size)(int *);
void FORTRAN(sampleij_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
       DFTYPE *, DFTYPE *, DFTYPE *);


void FORTRAN(samplei_multi_init)(int *);
void FORTRAN(samplei_multi_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(samplej_multi_init)(int *);
void FORTRAN(samplej_multi_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(samplek_multi_init)(int *);
void FORTRAN(samplek_multi_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(samplel_multi_init)(int *);
void FORTRAN(samplel_multi_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(samplem_multi_init)(int *);
void FORTRAN(samplem_multi_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(samplen_multi_init)(int *);
void FORTRAN(samplen_multi_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(samplet_date_init)(int *);
void FORTRAN(samplet_date_result_limits)(int *);
void FORTRAN(samplet_date_work_size)(int *);
void FORTRAN(samplet_date_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *);

void FORTRAN(samplef_date_init)(int *); 
void FORTRAN(samplef_date_result_limits)(int *);
void FORTRAN(samplef_date_work_size)(int *);
void FORTRAN(samplef_date_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *);


void FORTRAN(samplexy_init)(int *);
void FORTRAN(samplexy_result_limits)(int *);
void FORTRAN(samplexy_work_size)(int *);
void FORTRAN(samplexy_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexyt_init)(int *);
void FORTRAN(samplexyt_result_limits)(int *);
void FORTRAN(samplexyt_work_size)(int *);
void FORTRAN(samplexyt_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexyz_init)(int *);
void FORTRAN(samplexyz_result_limits)(int *);
void FORTRAN(samplexyz_work_size)(int *);
void FORTRAN(samplexyz_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexyzt_init)(int *);
void FORTRAN(samplexyzt_result_limits)(int *);
void FORTRAN(samplexyzt_work_size)(int *);
void FORTRAN(samplexyzt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexyt_nrst_init)(int *);
void FORTRAN(samplexyt_nrst_result_limits)(int *);
void FORTRAN(samplexyt_nrst_work_size)(int *);
void FORTRAN(samplexyt_nrst_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridgauss_xy_init)(int *);
void FORTRAN(scat2gridgauss_xy_work_size)(int *);
void FORTRAN(scat2gridgauss_xy_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridgauss_xz_init)(int *);
void FORTRAN(scat2gridgauss_xz_work_size)(int *);
void FORTRAN(scat2gridgauss_xz_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridgauss_yz_init)(int *);
void FORTRAN(scat2gridgauss_yz_work_size)(int *);
void FORTRAN(scat2gridgauss_yz_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridgauss_xt_init)(int *);
void FORTRAN(scat2gridgauss_xt_work_size)(int *);
void FORTRAN(scat2gridgauss_xt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridgauss_yt_init)(int *);
void FORTRAN(scat2gridgauss_yt_work_size)(int *);
void FORTRAN(scat2gridgauss_yt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridgauss_zt_init)(int *);
void FORTRAN(scat2gridgauss_zt_work_size)(int *);
void FORTRAN(scat2gridgauss_zt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridlaplace_xy_init)(int *);
void FORTRAN(scat2gridlaplace_xy_work_size)(int *);
void FORTRAN(scat2gridlaplace_xy_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridlaplace_xz_init)(int *);
void FORTRAN(scat2gridlaplace_xz_work_size)(int *);
void FORTRAN(scat2gridlaplace_xz_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridlaplace_yz_init)(int *);
void FORTRAN(scat2gridlaplace_yz_work_size)(int *);
void FORTRAN(scat2gridlaplace_yz_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *);


void FORTRAN(scat2gridlaplace_xt_init)(int *);
void FORTRAN(scat2gridlaplace_xt_work_size)(int *);
void FORTRAN(scat2gridlaplace_xt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridlaplace_yt_init)(int *);
void FORTRAN(scat2gridlaplace_yt_work_size)(int *);
void FORTRAN(scat2gridlaplace_yt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *);

void FORTRAN(scat2gridlaplace_zt_init)(int *);
void FORTRAN(scat2gridlaplace_zt_work_size)(int *);
void FORTRAN(scat2gridlaplace_zt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *);

void FORTRAN(sorti_init)(int *);
void FORTRAN(sorti_result_limits)(int *);
void FORTRAN(sorti_work_size)(int *);
void FORTRAN(sorti_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *);

void FORTRAN(sorti_str_init)(int *);
void FORTRAN(sorti_str_result_limits)(int *);
void FORTRAN(sorti_str_work_size)(int *);
void FORTRAN(sorti_str_compute)(int *, char *, DFTYPE *,
      char *, DFTYPE *);

void FORTRAN(sortj_init)(int *);
void FORTRAN(sortj_result_limits)(int *);
void FORTRAN(sortj_work_size)(int *);
void FORTRAN(sortj_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *);

void FORTRAN(sortj_str_init)(int *);
void FORTRAN(sortj_str_result_limits)(int *);
void FORTRAN(sortj_str_work_size)(int *);
void FORTRAN(sortj_str_compute)(int *, char *, DFTYPE *,
      char *, DFTYPE *);

void FORTRAN(sortk_init)(int *);
void FORTRAN(sortk_result_limits)(int *);
void FORTRAN(sortk_work_size)(int *);
void FORTRAN(sortk_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *);

void FORTRAN(sortk_str_init)(int *);
void FORTRAN(sortk_str_result_limits)(int *);
void FORTRAN(sortk_str_work_size)(int *);
void FORTRAN(sortk_str_compute)(int *, char *, DFTYPE *,
      char *, DFTYPE *);

void FORTRAN(sortl_init)(int *);
void FORTRAN(sortl_result_limits)(int *);
void FORTRAN(sortl_work_size)(int *);
void FORTRAN(sortl_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *);

void FORTRAN(sortl_str_init)(int *);
void FORTRAN(sortl_str_result_limits)(int *);
void FORTRAN(sortl_str_work_size)(int *);
void FORTRAN(sortl_str_compute)(int *, char *, DFTYPE *,
      char *, DFTYPE *);


void FORTRAN(sortm_init)(int *);
void FORTRAN(sortm_result_limits)(int *);
void FORTRAN(sortm_work_size)(int *);
void FORTRAN(sortm_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *);

void FORTRAN(sortm_str_init)(int *);
void FORTRAN(sortm_str_result_limits)(int *);
void FORTRAN(sortm_str_work_size)(int *);
void FORTRAN(sortm_str_compute)(int *, char *, DFTYPE *,
      char *, DFTYPE *);


void FORTRAN(sortn_init)(int *);
void FORTRAN(sortn_result_limits)(int *);
void FORTRAN(sortn_work_size)(int *);
void FORTRAN(sortn_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *);

void FORTRAN(sortn_str_init)(int *);
void FORTRAN(sortn_str_result_limits)(int *);
void FORTRAN(sortn_str_work_size)(int *);
void FORTRAN(sortn_str_compute)(int *, char *, DFTYPE *,
      char *, DFTYPE *);

void FORTRAN(tauto_cor_init)(int *);
void FORTRAN(tauto_cor_result_limits)(int *);
void FORTRAN(tauto_cor_work_size)(int *);
void FORTRAN(tauto_cor_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *);

void FORTRAN(xauto_cor_init)(int *);
void FORTRAN(xauto_cor_result_limits)(int *);
void FORTRAN(xauto_cor_work_size)(int *);
void FORTRAN(xauto_cor_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *);

void FORTRAN(eof_space_init)(int *);
void FORTRAN(eof_space_result_limits)(int *);
void FORTRAN(eof_space_work_size)(int *);
void FORTRAN(eof_space_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(eof_stat_init)(int *);
void FORTRAN(eof_stat_result_limits)(int *);
void FORTRAN(eof_stat_work_size)(int *);
void FORTRAN(eof_stat_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(eof_tfunc_init)(int *);
void FORTRAN(eof_tfunc_result_limits)(int *);
void FORTRAN(eof_tfunc_work_size)(int *);
void FORTRAN(eof_tfunc_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(eofsvd_space_init)(int *);
void FORTRAN(eofsvd_space_result_limits)(int *);
void FORTRAN(eofsvd_space_work_size)(int *);
void FORTRAN(eofsvd_space_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *);

void FORTRAN(eofsvd_stat_init)(int *);
void FORTRAN(eofsvd_stat_result_limits)(int *);
void FORTRAN(eofsvd_stat_work_size)(int *);
void FORTRAN(eofsvd_stat_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *);

void FORTRAN(eofsvd_tfunc_init)(int *);
void FORTRAN(eofsvd_tfunc_result_limits)(int *);
void FORTRAN(eofsvd_tfunc_work_size)(int *);
void FORTRAN(eofsvd_tfunc_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                           DFTYPE *);

void FORTRAN(compressi_init)(int *);
void FORTRAN(compressi_result_limits)(int *);
void FORTRAN(compressi_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressj_init)(int *);
void FORTRAN(compressj_result_limits)(int *);
void FORTRAN(compressj_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressk_init)(int *);
void FORTRAN(compressk_result_limits)(int *);
void FORTRAN(compressk_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressl_init)(int *);
void FORTRAN(compressl_result_limits)(int *);
void FORTRAN(compressl_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressm_init)(int *);
void FORTRAN(compressm_result_limits)(int *);
void FORTRAN(compressm_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressn_init)(int *);
void FORTRAN(compressn_result_limits)(int *);
void FORTRAN(compressn_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressi_by_init)(int *);
void FORTRAN(compressi_by_result_limits)(int *);
void FORTRAN(compressi_by_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressj_by_init)(int *);
void FORTRAN(compressj_by_result_limits)(int *);
void FORTRAN(compressj_by_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressk_by_init)(int *);
void FORTRAN(compressk_by_result_limits)(int *);
void FORTRAN(compressk_by_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressl_by_init)(int *);
void FORTRAN(compressl_by_result_limits)(int *);
void FORTRAN(compressl_by_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressm_by_init)(int *);
void FORTRAN(compressm_by_result_limits)(int *);
void FORTRAN(compressm_by_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(compressn_by_init)(int *);
void FORTRAN(compressn_by_result_limits)(int *);
void FORTRAN(compressn_by_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(box_edges_init)(int *);
void FORTRAN(box_edges_result_limits)(int *);
void FORTRAN(box_edges_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(labwid_init)(int *);
void FORTRAN(labwid_result_limits)(int *);
void FORTRAN(labwid_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(convolvei_init)(int *);
void FORTRAN(convolvei_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(convolvej_init)(int *);
void FORTRAN(convolvej_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(convolvek_init)(int *);
void FORTRAN(convolvek_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(convolvel_init)(int *);
void FORTRAN(convolvel_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(convolvem_init)(int *);
void FORTRAN(convolvem_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(convolven_init)(int *);
void FORTRAN(convolven_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(curv_range_init)(int *);
void FORTRAN(curv_range_result_limits)(int *);
void FORTRAN(curv_range_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(curv_to_rect_map_init)(int *);
void FORTRAN(curv_to_rect_map_result_limits)(int *);
void FORTRAN(curv_to_rect_map_work_size)(int *);
void FORTRAN(curv_to_rect_map_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                                       DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
void FORTRAN(curv_to_rect_init)(int *);
void FORTRAN(curv_to_rect_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(curv_to_rect_fsu_init)(int *);
void FORTRAN(curv_to_rect_fsu_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(rect_to_curv_init)(int *);
void FORTRAN(rect_to_curv_work_size)(int *);
void FORTRAN(rect_to_curv_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                                       DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(date1900_init)(int *);
void FORTRAN(date1900_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(days1900toymdhms_init)(int *);
void FORTRAN(days1900toymdhms_result_limits)(int *);
void FORTRAN(days1900toymdhms_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(minutes24_init)(int *);
void FORTRAN(minutes24_result_limits)(int *);
void FORTRAN(minutes24_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(element_index_init)(int *);
void FORTRAN(element_index_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(element_index_str_init)(int *);
void FORTRAN(element_index_str_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(element_index_str_n_init)(int *);
void FORTRAN(element_index_str_n_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(expnd_by_len_init)(int *);
void FORTRAN(expnd_by_len_result_limits)(int *);
void FORTRAN(expnd_by_len_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(expnd_by_len_str_init)(int *);
void FORTRAN(expnd_by_len_str_result_limits)(int *);
void FORTRAN(expnd_by_len_str_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(expndi_by_init)(int *);
void FORTRAN(expndi_by_result_limits)(int *);
void FORTRAN(expndi_by_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(expndi_by_t_init)(int *);
void FORTRAN(expndi_by_t_result_limits)(int *);
void FORTRAN(expndi_by_t_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(expndi_by_z_init)(int *);
void FORTRAN(expndi_by_z_result_limits)(int *);
void FORTRAN(expndi_by_z_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(expndi_by_z_counts_init)(int *);
void FORTRAN(expndi_by_z_counts_result_limits)(int *);
void FORTRAN(expndi_by_z_counts_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(expndi_id_by_z_counts_init)(int *);
void FORTRAN(expndi_id_by_z_counts_result_limits)(int *);
void FORTRAN(expndi_id_by_z_counts_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(expndi_by_m_counts_init)(int *);
void FORTRAN(expndi_by_m_counts_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(expndi_by_m_counts_str_init)(int *);
void FORTRAN(expndi_by_m_counts_str_compute)(int *, char, DFTYPE *, DFTYPE *, DFTYPE *, char);

void FORTRAN(fc_isubset_init)(int *);
void FORTRAN(fc_isubset_result_limits)(int *);
void FORTRAN(fc_isubset_custom_axes)(int *);
void FORTRAN(fc_isubset_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(findhi_init)(int *);
void FORTRAN(findhi_result_limits)(int *);
void FORTRAN(findhi_work_size)(int *);
void FORTRAN(findhi_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                            DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(findlo_init)(int *);
void FORTRAN(findlo_result_limits)(int *);
void FORTRAN(findlo_work_size)(int *);
void FORTRAN(findlo_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                            DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(is_element_of_init)(int *);
void FORTRAN(is_element_of_result_limits)(int *);
void FORTRAN(is_element_of_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(is_element_of_str_init)(int *);
void FORTRAN(is_element_of_str_result_limits)(int *);
void FORTRAN(is_element_of_str_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);


void FORTRAN(is_element_of_str_n_init)(int *);
void FORTRAN(is_element_of_str_n_result_limits)(int *);
void FORTRAN(is_element_of_str_n_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(lanczos_init)(int *);
void FORTRAN(lanczos_work_size)(int *);
void FORTRAN(lanczos_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                            DFTYPE *, DFTYPE *);

void FORTRAN(lsl_lowpass_init)(int *);
void FORTRAN(lsl_lowpass_work_size)(int *);
void FORTRAN(lsl_lowpass_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
                            DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexy_curv_init)(int *);
void FORTRAN(samplexy_curv_result_limits)(int *);
void FORTRAN(samplexy_curv_work_size)(int *);
void FORTRAN(samplexy_curv_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexy_curv_avg_init)(int *);
void FORTRAN(samplexy_curv_avg_result_limits)(int *);
void FORTRAN(samplexy_curv_avg_work_size)(int *);
void FORTRAN(samplexy_curv_avg_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexy_curv_nrst_init)(int *);
void FORTRAN(samplexy_curv_nrst_result_limits)(int *);
void FORTRAN(samplexy_curv_nrst_work_size)(int *);
void FORTRAN(samplexy_curv_nrst_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexy_closest_init)(int *);
void FORTRAN(samplexy_closest_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexy_nrst_init)(int *);
void FORTRAN(samplexy_nrst_result_limits)(int *);
void FORTRAN(samplexy_nrst_work_size)(int *);
void FORTRAN(samplexy_nrst_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(samplexz_init)(int *);
void FORTRAN(samplexz_result_limits)(int *);
void FORTRAN(samplexz_work_size)(int *);
void FORTRAN(samplexz_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(sampleyz_init)(int *);
void FORTRAN(sampleyz_result_limits)(int *);
void FORTRAN(sampleyz_work_size)(int *);
void FORTRAN(sampleyz_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2ddups_init)(int *);
void FORTRAN(scat2ddups_result_limits)(int *);
void FORTRAN(scat2ddups_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(ave_scat2grid_t_init)(int *);
void FORTRAN(ave_scat2grid_t_work_size)(int *);
void FORTRAN(ave_scat2grid_t_compute)(int *, DFTYPE *, DFTYPE *,
      DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_t_init)(int *);
void FORTRAN(scat2grid_t_work_size)(int *);
void FORTRAN(scat2grid_t_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_ef_init)(int *);
void FORTRAN(transpose_ef_result_limits)(int *);
void FORTRAN(transpose_ef_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_te_init)(int *);
void FORTRAN(transpose_te_result_limits)(int *);
void FORTRAN(transpose_te_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_tf_init)(int *);
void FORTRAN(transpose_tf_result_limits)(int *);
void FORTRAN(transpose_tf_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_xe_init)(int *);
void FORTRAN(transpose_xe_result_limits)(int *);
void FORTRAN(transpose_xe_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_xf_init)(int *);
void FORTRAN(transpose_xf_result_limits)(int *);
void FORTRAN(transpose_xf_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_xt_init)(int *);
void FORTRAN(transpose_xt_result_limits)(int *);
void FORTRAN(transpose_xt_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_xy_init)(int *);
void FORTRAN(transpose_xy_result_limits)(int *);
void FORTRAN(transpose_xy_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_xz_init)(int *);
void FORTRAN(transpose_xz_result_limits)(int *);
void FORTRAN(transpose_xz_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_ye_init)(int *);
void FORTRAN(transpose_ye_result_limits)(int *);
void FORTRAN(transpose_ye_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_yf_init)(int *);
void FORTRAN(transpose_yf_result_limits)(int *);
void FORTRAN(transpose_yf_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_yt_init)(int *);
void FORTRAN(transpose_yt_result_limits)(int *);
void FORTRAN(transpose_yt_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_yz_init)(int *);
void FORTRAN(transpose_yz_result_limits)(int *);
void FORTRAN(transpose_yz_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_ze_init)(int *);
void FORTRAN(transpose_ze_result_limits)(int *);
void FORTRAN(transpose_ze_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_zf_init)(int *);
void FORTRAN(transpose_zf_result_limits)(int *);
void FORTRAN(transpose_zf_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(transpose_zt_init)(int *);
void FORTRAN(transpose_zt_result_limits)(int *);
void FORTRAN(transpose_zt_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(xcat_init)(int *);
void FORTRAN(xcat_result_limits)(int *);
void FORTRAN(xcat_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(xcat_str_init)(int *);
void FORTRAN(xcat_str_result_limits)(int *);
void FORTRAN(xcat_str_compute)(int *, char *, char *, char *);

void FORTRAN(ycat_init)(int *);
void FORTRAN(ycat_result_limits)(int *);
void FORTRAN(ycat_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(ycat_str_init)(int *);
void FORTRAN(ycat_str_result_limits)(int *);
void FORTRAN(ycat_str_compute)(int *, char *, char *, char *);

void FORTRAN(zcat_init)(int *);
void FORTRAN(zcat_result_limits)(int *);
void FORTRAN(zcat_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(zcat_str_init)(int *);
void FORTRAN(zcat_str_result_limits)(int *);
void FORTRAN(zcat_str_compute)(int *, char *, char *, char *);

void FORTRAN(tcat_init)(int *);
void FORTRAN(tcat_result_limits)(int *);
void FORTRAN(tcat_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tcat_str_init)(int *);
void FORTRAN(tcat_str_result_limits)(int *);
void FORTRAN(tcat_str_compute)(int *, char *, char *, char *);

void FORTRAN(ecat_init)(int *);
void FORTRAN(ecat_result_limits)(int *);
void FORTRAN(ecat_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(ecat_str_init)(int *);
void FORTRAN(ecat_str_result_limits)(int *);
void FORTRAN(ecat_str_compute)(int *, char *, char *, char *);

void FORTRAN(fcat_init)(int *);
void FORTRAN(fcat_result_limits)(int *);
void FORTRAN(fcat_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(fcat_str_init)(int *);
void FORTRAN(fcat_str_result_limits)(int *);
void FORTRAN(fcat_str_compute)(int *, char *, char *, char *);

void FORTRAN(xreverse_init)(int *);
void FORTRAN(xreverse_result_limits)(int *);
void FORTRAN(xreverse_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(yreverse_init)(int *);
void FORTRAN(yreverse_result_limits)(int *);
void FORTRAN(yreverse_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(zreverse_init)(int *);
void FORTRAN(zreverse_result_limits)(int *);
void FORTRAN(zreverse_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(treverse_init)(int *);
void FORTRAN(treverse_result_limits)(int *);
void FORTRAN(treverse_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(ereverse_init)(int *);
void FORTRAN(ereverse_result_limits)(int *);
void FORTRAN(ereverse_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(freverse_init)(int *);
void FORTRAN(freverse_result_limits)(int *);
void FORTRAN(freverse_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(zaxreplace_avg_init)(int *);
void FORTRAN(zaxreplace_avg_work_size)(int *);
void FORTRAN(zaxreplace_avg_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(zaxreplace_bin_init)(int *);
void FORTRAN(zaxreplace_bin_work_size)(int *);
void FORTRAN(zaxreplace_bin_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(zaxreplace_rev_init)(int *);
void FORTRAN(zaxreplace_rev_work_size)(int *);
void FORTRAN(zaxreplace_rev_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(zaxreplace_zlev_init)(int *);
void FORTRAN(zaxreplace_zlev_work_size)(int *);
void FORTRAN(zaxreplace_zlev_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(nco_attr_init)(int *);
void FORTRAN(nco_attr_result_limits)(int *);
void FORTRAN(nco_attr_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(nco_init)(int *);
void FORTRAN(nco_result_limits)(int *);
void FORTRAN(nco_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_datestring_init)(int *);
void FORTRAN(tax_datestring_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_day_init)(int *);
void FORTRAN(tax_day_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_dayfrac_init)(int *);
void FORTRAN(tax_dayfrac_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_jday1900_init)(int *);
void FORTRAN(tax_jday1900_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_jday_init)(int *);
void FORTRAN(tax_jday_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_month_init)(int *);
void FORTRAN(tax_month_work_size)(int *);
void FORTRAN(tax_month_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_times_init)(int *);
void FORTRAN(tax_times_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_tstep_init)(int *);
void FORTRAN(tax_tstep_work_size)(int *);
void FORTRAN(tax_tstep_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_units_init)(int *);
void FORTRAN(tax_units_compute)(int *, DFTYPE *, DFTYPE*);

void FORTRAN(tax_year_init)(int *);
void FORTRAN(tax_year_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tax_yearfrac_init)(int *);
void FORTRAN(tax_yearfrac_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(fill_xy_init)(int *);
void FORTRAN(fill_xy_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(test_opendap_init)(int *);
void FORTRAN(test_opendap_result_limits)(int *);
void FORTRAN(test_opendap_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_bin_xy_init)(int *);
void FORTRAN(scat2grid_bin_xy_work_size)(int *);
void FORTRAN(scat2grid_bin_xy_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_bin_xyt_init)(int *);
void FORTRAN(scat2grid_bin_xyt_work_size)(int *);
void FORTRAN(scat2grid_bin_xyt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_bin_xyz_init)(int *);
void FORTRAN(scat2grid_bin_xyz_work_size)(int *);
void FORTRAN(scat2grid_bin_xyz_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_bin_xyzt_init)(int *);
void FORTRAN(scat2grid_bin_xyzt_work_size)(int *);
void FORTRAN(scat2grid_bin_xyzt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_nbin_xy_init)(int *);
void FORTRAN(scat2grid_nbin_xy_work_size)(int *);
void FORTRAN(scat2grid_nbin_xy_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_nbin_xyt_init)(int *);
void FORTRAN(scat2grid_nbin_xyt_work_size)(int *);
void FORTRAN(scat2grid_nbin_xyt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_nobs_xyt_init)(int *);
void FORTRAN(scat2grid_nobs_xyt_work_size)(int *);
void FORTRAN(scat2grid_nobs_xyt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_nobs_xy_init)(int *);
void FORTRAN(scat2grid_nobs_xy_work_size)(int *);
void FORTRAN(scat2grid_nobs_xy_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(unique_str2int_init)(int *);
void FORTRAN(unique_str2int_compute)(char *, int *);

void FORTRAN(bin_index_wt_init)(int *);
void FORTRAN(bin_index_wt_result_limits)(int *);
void FORTRAN(bin_index_wt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(minmax_init)(int *);
void FORTRAN(minmax_result_limits)(int *);
void FORTRAN(minmax_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(floatstr_init)(int *);
void FORTRAN(floatstr_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(pt_in_poly_init)(int *);
void FORTRAN(pt_in_poly_work_size)(int *);
void FORTRAN(pt_in_poly_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(list_value_xml_init)(int *);
void FORTRAN(list_value_xml_result_limits)(int *);
void FORTRAN(list_value_xml_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(lon_lat_time_string_init)(int *);
void FORTRAN(lon_lat_time_string_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, char *);

void FORTRAN(write_webrow_init)(int *);
void FORTRAN(write_webrow_result_limits)(int *);
void FORTRAN(write_webrow_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(str_mask_init)(int *);
void FORTRAN(str_mask_compute)(int *, DFTYPE *, DFTYPE *);

void FORTRAN(separate_init)(int *);
void FORTRAN(separate_result_limits)(int *);
void FORTRAN(separate_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(time_reformat_init)(int *);
void FORTRAN(time_reformat_compute)(int *, char *);

void FORTRAN(ft_to_orthogonal_init)(int *);
void FORTRAN(ft_to_orthogonal_work_size)(int *);
void FORTRAN(ft_to_orthogonal_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(separate_str_init)(int *);
void FORTRAN(separate_str_result_limits)(int *);
void FORTRAN(separate_str_compute)(int *, char *, DFTYPE *, DFTYPE *, char *);

void FORTRAN(sample_fast_i_init)(int *);
void FORTRAN(sample_fast_i_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(sample_fast_i_str_init)(int *);
void FORTRAN(sample_fast_i_str_compute)(int *, char *, DFTYPE *, char *);

void FORTRAN(piecewise3_init)(int *);
void FORTRAN(piecewise3_result_limits)(int *);
void FORTRAN(piecewise3_work_size)(int *);
void FORTRAN(piecewise3_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(write_webrow_gwt_init)(int *);
void FORTRAN(write_webrow_gwt_result_limits)(int *);
void FORTRAN(write_webrow_gwt_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(str_noblanks_init)(int *);
void FORTRAN(str_noblanks_compute)(int *, char *, char *);

void FORTRAN(str_replace_init)(int *);
void FORTRAN(str_replace_compute)(int *, char *, char *, char *, char *);

void FORTRAN(expndi_to_et_init)(int *);
void FORTRAN(expndi_to_et_work_size)(int *);
void FORTRAN(expndi_to_et_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(dot_x_init)(int *);
void FORTRAN(dot_x_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(dot_y_init)(int *);
void FORTRAN(dot_y_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(dot_z_init)(int *);
void FORTRAN(dot_z_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(dot_t_init)(int *);
void FORTRAN(dot_t_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(dot_e_init)(int *);
void FORTRAN(dot_e_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(dot_f_init)(int *);
void FORTRAN(dot_f_compute)(int *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(tracks2grid_mask_ave_xyt_init)(int *);
void FORTRAN(tracks2grid_mask_ave_xyt_work_size)(int *);
void FORTRAN(tracks2grid_mask_ave_xyt_compute)(int *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN( tracks2grid_std_xyt_init)(int *);
void FORTRAN( tracks2grid_std_xyt_work_size)(int *);
void FORTRAN( tracks2grid_std_xyt_compute)(int *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

void FORTRAN(scat2grid_minmax_xyt_init)(int *);
void FORTRAN(scat2grid_minmax_xyt_result_limits)(int *);
void FORTRAN(scat2grid_minmax_xyt_work_size)(int *);
void FORTRAN(scat2grid_minmax_xyt_compute)(int *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);


void FORTRAN( scat2grid_std_xyt_init)(int *);
void FORTRAN( scat2grid_std_xyt_work_size)(int *);
void FORTRAN( scat2grid_std_xyt_compute)(int *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
  DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);


void FORTRAN( earth_distance_init)(int *);
void FORTRAN( earth_distance_compute)(int *,
  DFTYPE *, DFTYPE *, DFTYPE *);

/*
 *  End of declarations for internally linked external functions
 *  ------------------------------------ */


/* .............. Function Definitions .............. */


/* .... Functions for use by Ferret (to be called from Fortran) .... */

/*
 * Note that all routines called directly from Ferret,
 * ie. directly from Fortran, should be all lower case,
 * should pass by reference and should end with an underscore.
 */

static int continue_efcn_scan(int gfcn_num_internal);

/*
 * Find all of the ~.so files in directories listed in the
 * PYFER_EXTERNAL_FUNCTIONS environment variable and add all 
 * the names and associated directory information to the 
 * STATIC_ExternalFunctionList.
 */
int FORTRAN(efcn_scan)( int *gfcn_num_internal )
{
  int return_val;

  /* Called multiple times but only do the setup once. */
  if ( I_have_scanned_already ) {
    return_val = list_size(STATIC_ExternalFunctionList);
    return return_val;
  }

  return_val = continue_efcn_scan(*gfcn_num_internal);
  if ( return_val >= 0 )
      I_have_scanned_already = TRUE;
  return return_val;
}

/*
 * Split out as a separate function to minimize cost of calls after scanned.
 */
static int continue_efcn_scan(int gfcn_num_internal) {
  FILE *file_ptr=NULL;
  ExternalFunction ef;

  char file[EF_MAX_NAME_LENGTH]="";
  char *path_ptr=NULL;
  char path[8192]="";
  char allpaths[8192]="";
  char cmd[8192]="";   /* EF_MAX_DESCRIPTION_LENGTH */
  int  count=0;
  int  i_intEF;
  char *extension;

  /* The array of names of the internally defined "external functions" defined here. */
  const char I_EFnames[][EF_MAX_NAME_LENGTH] = {
     "ave_scat2grid_t",
     "bin_index_wt",
     "box_edges",
     "compressi",
     "compressi_by",
     "compressj",
     "compressj_by",
     "compressk",
     "compressk_by",
     "compressl",
     "compressl_by",
     "compressm",
     "compressm_by",
     "compressn",
     "compressn_by",
     "convolvei",
     "convolvej",
     "convolvek",
     "convolvel",
     "convolvem",
     "convolven",
     "curv_range",
     "curv_to_rect",
     "curv_to_rect_fsu",
     "curv_to_rect_map",
     "date1900",
     "days1900toymdhms",
     "dot_e",
     "dot_f",
     "dot_t",
     "dot_x",
     "dot_y",
     "dot_z",
     "dsg_fmask_str",
     "earth_distance",
     "ecat",
     "ecat_str",
     "element_index",
     "element_index_str",
     "element_index_str_n",
     "eof_space",
     "eof_stat",
     "eofsvd_space",
     "eofsvd_stat",
     "eofsvd_tfunc",
     "eof_tfunc",
     "ereverse",
     "expnd_by_len",
     "expnd_by_len_str",
     "expndi_by",
     "expndi_by_m_counts",
     "expndi_by_m_counts_str",
     "expndi_by_t",
     "expndi_by_z",
     "expndi_by_z_counts",
     "expndi_id_by_z_counts",
     "expndi_to_et",
     "fcat",
     "fcat_str",
     "fc_isubset",
     "ffta",
     "fft_im",
     "fft_inverse",
     "fftp",
     "fft_re",
     "fill_xy",
     "findhi",
     "findlo",
     "floatstr",
     "freverse",
     "ft_to_orthogonal",
     "is_element_of",
     "is_element_of_str",
     "is_element_of_str_n",
     "labwid",
     "lanczos",
     "list_value_xml",
     "lon_lat_time_string",
     "lsl_lowpass",
     "minmax",
     "minutes24",
     "nco",
     "nco_attr",
     "piecewise3",
     "pt_in_poly",
     "rect_to_curv",
     "sample_fast_i",
     "sample_fast_i_str",
     "samplef_date",
     "sampleij",
     "samplei_multi",
     "samplej_multi",
     "samplek_multi",
     "samplel_multi",
     "samplem_multi",
     "samplen_multi",
     "samplet_date",
     "samplexy",
     "samplexy_closest",
     "samplexy_curv",
     "samplexy_curv_avg",
     "samplexy_curv_nrst",
     "samplexy_nrst",
     "samplexyt_nrst",
     "samplexyt",
     "samplexyz",
     "samplexyzt",
     "samplexz",
     "sampleyz",
     "scat2ddups",
     "scat2grid_bin_xy",
     "scat2grid_bin_xyt",
     "scat2grid_bin_xyz",
     "scat2grid_bin_xyzt",
     "scat2grid_minmax_xyt",
     "scat2grid_std_xyt",
     "scat2gridgauss_xt",
     "scat2gridgauss_xy",
     "scat2gridgauss_xz",
     "scat2gridgauss_yt",
     "scat2gridgauss_yz",
     "scat2gridgauss_zt",
     "scat2gridlaplace_xt",
     "scat2gridlaplace_xy",
     "scat2gridlaplace_xz",
     "scat2gridlaplace_yt",
     "scat2gridlaplace_yz",
     "scat2gridlaplace_zt",
     "scat2grid_nbin_xy",
     "scat2grid_nbin_xyt",
     "scat2grid_nobs_xy",
     "scat2grid_nobs_xyt",
     "scat2grid_t",
     "separate",
     "separate_str",
     "sorti",
     "sorti_str",
     "sortj",
     "sortj_str",
     "sortk",
     "sortk_str",
     "sortl",
     "sortl_str",
     "sortm",
     "sortm_str",
     "sortn",
     "sortn_str",
     "str_mask",
     "str_noblanks",
     "str_replace",
     "tauto_cor",
     "tax_datestring",
     "tax_day",
     "tax_dayfrac",
     "tax_jday",
     "tax_jday1900",
     "tax_month",
     "tax_times",
     "tax_tstep",
     "tax_units",
     "tax_year",
     "tax_yearfrac",
     "tcat",
     "tcat_str",
     "test_opendap",
     "time_reformat",
     "tracks2grid_mask_ave_xyt",
     "tracks2grid_std_xyt",
     "transpose_ef",
     "transpose_te",
     "transpose_tf",
     "transpose_xe",
     "transpose_xf",
     "transpose_xt",
     "transpose_xy",
     "transpose_xz",
     "transpose_ye",
     "transpose_yf",
     "transpose_yt",
     "transpose_yz",
     "transpose_ze",
     "transpose_zf",
     "transpose_zt",
     "treverse",
     "unique_str2int",
     "write_webrow",
     "write_webrow_gwt",
     "xauto_cor",
     "xcat",
     "xcat_str",
     "xreverse",
     "ycat",
     "ycat_str",
     "yreverse",
     "zaxreplace_avg",
     "zaxreplace_bin",
     "zaxreplace_rev",
     "zaxreplace_zlev",
     "zcat",
     "zcat_str",
     "zreverse"
  };
  /* The number of names in the array above */
  int N_INTEF = sizeof(I_EFnames) / EF_MAX_NAME_LENGTH;

  if ( (STATIC_ExternalFunctionList = list_init(__FILE__, __LINE__)) == NULL ) {
    fputs("**ERROR: efcn_scan: Unable to initialize STATIC_ExternalFunctionList.\n", stderr);
    return -1;
  }

  /*
   * Get internally linked external functions;  and add all
   * the names and associated directory information to the
   * STATIC_ExternalFunctionList.
   */
  for (i_intEF = 0; i_intEF < N_INTEF; i_intEF++ ) {
      strcpy(ef.path, "internally_linked");
      strcpy(ef.name, I_EFnames[i_intEF]);
      ef.id = gfcn_num_internal + ++count; /* pre-increment because F arrays start at 1 */
      ef.already_have_internals = NO;
      ef.internals_ptr = NULL;
      list_insert_after(STATIC_ExternalFunctionList, (char *) &ef, sizeof(ExternalFunction), __FILE__, __LINE__);
  }

  /*
   * - Get all the paths from the "PYFER_EXTERNAL_FUNCTIONS" environment variable.
   *
   * - While there is another path:
   *    - get the path;
   *    - create a pipe for the "ls -1" command;
   *    - read stdout and use each file name to create another external function entry;
   *
   */

  path_ptr = getenv("PYFER_EXTERNAL_FUNCTIONS");
  if ( path_ptr == NULL ) {
    /* fprintf(stderr, "\nWARNING: environment variable PYFER_EXTERNAL_FUNCTIONS not defined.\n\n"); */
    return count;
  }

  strcpy(allpaths, path_ptr);
  path_ptr = strtok(allpaths, " \t");
  if ( path_ptr == NULL ) {

    /* fprintf(stderr, "\nWARNING:No paths were found in the environment variable PYFER_EXTERNAL_FUNCTIONS.\n\n"); */
    return count;

  } else {

    do {
      strcpy(path, path_ptr);
      if (path[strlen(path)-1] != '/')
        strcat(path, "/");

      sprintf(cmd, "ls -1 %s", path);

      /* Open a pipe to the "ls" command */
      if ( (file_ptr = popen(cmd, "r")) == (FILE *) NULL ) {
         fputs("**ERROR: Cannot open pipe.\n", stderr);
         return -1;
      }

      /*
       * Read a line at a time.
       * Any ~.so files are assumed to be external functions.
       */
      while ( fgets(file, EF_MAX_NAME_LENGTH, file_ptr) != NULL ) {

         extension = &(file[strlen(file)-1]);
         while ( isspace(*extension) ) {
            *extension = '\0';   /* chop off the carriage return (or CR-LF) */
            extension--;
         }
         extension--;
         extension--;
         if ( strcmp(extension, ".so") == 0 ) {
            *extension = '\0'; /* chop off the ".so" */
            strcpy(ef.path, path);
            strcpy(ef.name, file);
            ef.id = gfcn_num_internal + ++count; /* pre-increment because F arrays start at 1 */
            ef.already_have_internals = NO;
            ef.internals_ptr = NULL;
            list_insert_after(STATIC_ExternalFunctionList, (char *) &ef, sizeof(ExternalFunction), __FILE__, __LINE__);
         }
      }

      pclose(file_ptr);

      path_ptr = strtok(NULL, " \t"); /* get the next directory */
    } while ( path_ptr != NULL );

  }

  return count;
}

/*
 * Clears and frees all memory associated with the given ef pointed to by data
 */
static void efcn_dealloc_ef(char *data) {
  ExternalFunction *ef_ptr = (ExternalFunction *)data;
  if ( ef_ptr->internals_ptr != NULL ) {
      /* paranoia */
      memset(ef_ptr->internals_ptr, 0, sizeof(ExternalFunctionInternals));
      FerMem_Free(ef_ptr->internals_ptr, __FILE__, __LINE__);
  }
  /* paranoia */
  memset(ef_ptr, 0, sizeof(ExternalFunction));
  FerMem_Free(ef_ptr, __FILE__, __LINE__);
}

void FORTRAN(efcn_list_clear)(void)
{
  if ( STATIC_ExternalFunctionList != NULL ) {
      /* free all the elements in the list and the list itseld */
      list_free(STATIC_ExternalFunctionList, efcn_dealloc_ef, __FILE__, __LINE__);
      STATIC_ExternalFunctionList = NULL;
  }
  I_have_scanned_already = FALSE;
}

/*
 * Determine whether an external function has already
 * had its internals read.
 */
int FORTRAN(efcn_already_have_internals)( int *id_ptr )
{
  ExternalFunction *ef_ptr;
  int return_val;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     return 0;
  }

  return_val = ef_ptr->already_have_internals;
  return return_val;
}



/*
 * Create a new python-backed external function.  The initialization of
 * this function is done at this time to ensure that the python module is
 * valid and contains suitable functions.  Initialization is accomplished
 * using generic wrapper functions.
 * Input arguments:
 *    fname - name for the function
 *    lenfname - actual length of the name in fname
 *    pymod - name of the python module suitable for a python import statement
 *            (eg, "package.subpackage.module")
 *    lenpymod - actual length of the name in pymod
 * Output arguments:
 *    errstring - error message if something went wrong
 *    lenerrstring - actual length of the string returned in errstring
 * The value of lenerrstring will be zero if and only if there were no errors
 *
 * Note: this function assume Hollerith strings are passed as character arrays
 *       (and max lengths appended as ints to the end of the argument list -
 *        they are not listed here since unused; also permits saying the strings
 *        are simple arrays in Fortran)
 */
void FORTRAN(create_pyefcn)(char fname[], int *lenfname, char pymod[], int *lenpymod,
                            char errstring[], int *lenerrstring)
{
    ExternalFunction ef;
    ExternalFunction *ef_ptr;

    /* Check string lengths since these values might possibly be exceeded */
    if ( *lenpymod >= EF_MAX_DESCRIPTION_LENGTH ) {
        sprintf(errstring, "Module name too long (must be less than %d characters)", EF_MAX_DESCRIPTION_LENGTH);
        *lenerrstring = strlen(errstring);
        return;
    }
    if ( *lenfname >= EF_MAX_NAME_LENGTH ) {
        sprintf(errstring, "Function name too long (must be less than %d characters)", EF_MAX_NAME_LENGTH);
        *lenerrstring = strlen(errstring);
        return;
    }

    /*
     * Assign the local ExternalFunction structure, assigning the module name to the path element
     * Get the ID for this new function by adding one to the ID of the last element in the list.
     * (The IDs do not match the size of the list.)
     */
    ef.handle = NULL;
    ef_ptr = (ExternalFunction *) list_rear(STATIC_ExternalFunctionList);
    ef.id = ef_ptr->id + 1;
    strncpy(ef.name, fname, *lenfname);
    ef.name[*lenfname] = '\0';
    strncpy(ef.path, pymod, *lenpymod);
    ef.path[*lenpymod] = '\0';
    ef.already_have_internals = FALSE;
    ef.internals_ptr = NULL;

    /* Add a copy of this ExternalFunction to the end of the global list of external functions */
    list_mvrear(STATIC_ExternalFunctionList);
    ef_ptr = (ExternalFunction *)list_insert_after(STATIC_ExternalFunctionList, (char *) &ef, sizeof(ExternalFunction), __FILE__, __LINE__);

    /* Allocate and initialize the internals data for this ExternalFunction in the list */
    if ( EF_New(ef_ptr) != 0 ) {
        strcpy(errstring, "Unable to allocate memory for the internals data in create_pyefcn");
        *lenerrstring = strlen(errstring);
        return;
    }
    ef_ptr->internals_ptr->language = EF_PYTHON;
    ef_ptr->already_have_internals = TRUE;

    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack
     * environment with sigsetjmp (for the signal handler) and setjmp
     * (for the "bail out" utility function).
     */
    if ( EF_Util_setsig("create_pyefcn")) {
        list_remove_rear(STATIC_ExternalFunctionList, __FILE__, __LINE__);
        FerMem_Free(ef_ptr->internals_ptr, __FILE__, __LINE__);
        FerMem_Free(ef_ptr, __FILE__, __LINE__);
        strcpy(errstring, "Unable to set signal handlers in create_pyefcn");
        *lenerrstring = strlen(errstring);
        return;
    }
    if (sigsetjmp(sigjumpbuffer, 1) != 0) {
        list_remove_rear(STATIC_ExternalFunctionList, __FILE__, __LINE__);
        FerMem_Free(ef_ptr->internals_ptr, __FILE__, __LINE__);
        FerMem_Free(ef_ptr, __FILE__, __LINE__);
        strcpy(errstring, "Signal caught in create_pyefcn");
        *lenerrstring = strlen(errstring);
        return;
    }
    if (setjmp(jumpbuffer) != 0) {
        list_remove_rear(STATIC_ExternalFunctionList, __FILE__, __LINE__);
        FerMem_Free(ef_ptr->internals_ptr, __FILE__, __LINE__);
        FerMem_Free(ef_ptr, __FILE__, __LINE__);
        strcpy(errstring, "ef_bail_out called in create_pyefcn");
        *lenerrstring = strlen(errstring);
        return;
    }
    canjump = 1;

    pyefcn_init(ef_ptr->id, ef_ptr->path, errstring);

    /* Restore the old signal handlers. */
    EF_Util_ressig("create_pyefcn");

    *lenerrstring = strlen(errstring);
    if ( *lenerrstring > 0 ) {
        list_remove_rear(STATIC_ExternalFunctionList, __FILE__, __LINE__);
        FerMem_Free(ef_ptr->internals_ptr, __FILE__, __LINE__);
        FerMem_Free(ef_ptr, __FILE__, __LINE__);
    }
    return;
}



/*
 * Find an external function based on its integer ID and
 * gather information describing the function.
 *
 * Return values:
 *     -1: error occurred, dynamic linking was unsuccessful
 *      0: success
 */
int FORTRAN(efcn_gather_info)( int *id_ptr )
{
  ExternalFunction *ef_ptr;
  int internally_linked;
  char tempText[1024];
  ExternalFunctionInternals *i_ptr;
  void (*f_init_ptr)(int *);

   /*
    * Find the external function.
    */
   ef_ptr = ef_ptr_from_id_ptr(id_ptr);
   if ( ef_ptr == NULL ) {
      fprintf(stderr, "**ERROR: No external function of id %d was found.\n", *id_ptr);
      return -1;
   }
   /* Check if this has already been done */
   if (ef_ptr->already_have_internals)  {
      return 0;
   }
   /* Check if this is an internal function */
   if ( strcmp(ef_ptr->path,"internally_linked") == 0 )
      internally_linked = TRUE;
   else
      internally_linked = FALSE;

   /* Get a handle for the shared object if not internally linked */
   if ( ! internally_linked ) {
      strcpy(tempText, ef_ptr->path);
      strcat(tempText, ef_ptr->name);
      strcat(tempText, ".so");

      if ( (ef_ptr->handle = dlopen(tempText, RTLD_LAZY)) == NULL ) {
         fprintf(stderr, "**ERROR in External Function %s:\n"
                         "  Dynamic linking call dlopen() returns --\n"
                         "  \"%s\".\n", ef_ptr->name, dlerror());
         return -1;
      }
   }

   /* Allocate and default initialize the internal information. */
   if ( EF_New(ef_ptr) != 0 )
      return -1;

   /* Call the external function initialization routine */
   i_ptr = ef_ptr->internals_ptr;

   if ( i_ptr->language == EF_F ) {

      /*
       * Prepare for bailout possibilities by setting a signal handler for
       * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack
       * environment with sigsetjmp (for the signal handler) and setjmp
       * (for the "bail out" utility function).
       */
      if ( EF_Util_setsig("efcn_gather_info")) {
         return -1;
      }

      /* Set the signal return location and process jumps */
      if ( sigsetjmp(sigjumpbuffer, 1) != 0 ) {
         /* Must have come from bailing out */
         return -1;
      }
      /* Set the bail out return location and process jumps */
      if ( setjmp(jumpbuffer) != 0 ) {
         /* Must have come from bailing out */
         return -1;
      }
      canjump = 1;

      /* Get the pointer to external function initialization routine */
      sprintf(tempText, "%s_init_", ef_ptr->name);
      if ( ! internally_linked ) {
         f_init_ptr = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
      } else {
         f_init_ptr = (void (*)(int *))internal_dlsym(tempText);
      }
      if ( f_init_ptr == NULL ) {
         fprintf(stderr, "**ERROR in efcn_gather_info(): %s is not found.\n", tempText);
         if ( ! internally_linked )
            fprintf(stderr, "  dlerror: \"%s\"\n", dlerror());
         EF_Util_ressig("efcn_gather_info");
         return -1;
      }

      /*
       * Call the initialization routine.  If it bails out,
       * this will jump back to one of the setjmp methods, returning non-zero.
       */
      (*f_init_ptr)(id_ptr);
      ef_ptr->already_have_internals = TRUE;

      /* Restore the old signal handlers. */
      if ( EF_Util_ressig("efcn_gather_info") ) {
         return -1;
      }

   }
   else {
      /* Note: Python-backed external functions get initialized when added, so no support here for them */
      fprintf(stderr, "**ERROR: unsupported language (%d) for efcn_gather_info.\n", i_ptr->language);
      return -1;
   }

   return 0;
}


/*
 * Find an external function based on its integer ID,
 * Query the function about custom axes. Store the context
 * list information for use by utility functions.
 */
void FORTRAN(efcn_get_custom_axes)( int *id_ptr, int *cx_list_ptr, int *status )
{
  ExternalFunction *ef_ptr=NULL;
  char tempText[EF_MAX_NAME_LENGTH]="";
  int internally_linked = FALSE;

  void (*fptr)(int *);

  /*
   * Initialize the status
   */
  *status = FERR_OK;

  /*
   * Store the context list globally.
   */
  EF_store_globals(NULL, cx_list_ptr, NULL, NULL);

  /*
   * Find the external function.
   */
  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  if ( (!strcmp(ef_ptr->path,"internally_linked")) ) {internally_linked = TRUE; }

  if ( ef_ptr->internals_ptr->language == EF_F ) {

    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack
     * environment with sigsetjmp (for the signal handler) and setjmp
     * (for the "bail out" utility function).
     */

    if (EF_Util_setsig("efcn_get_custom_axes")) {
      *status = FERR_EF_ERROR;
       return;
    }

    /*
     * Set the signal return location and process jumps
     */
    if (sigsetjmp(sigjumpbuffer, 1) != 0) {
      *status = FERR_EF_ERROR;
      return;
    }

    /*
     * Set the bail out return location and process jumps
     */
    if (setjmp(jumpbuffer) != 0) {
      *status = FERR_EF_ERROR;
      return;
    }

    canjump = 1;

    strcpy(tempText, ef_ptr->name);
    strcat(tempText, "_custom_axes_");

    if (!internally_linked) {
       fptr  = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
    } else {
      fptr  = (void (*)(int *))internal_dlsym(tempText);
    }
    (*fptr)( id_ptr );


    /*
     * Restore the old signal handlers.
     */
    if ( EF_Util_ressig("efcn_get_custom_axes")) {
       return;
    }

    /* end of EF_F */
  }
  else if ( ef_ptr->internals_ptr->language == EF_PYTHON ) {
      char errstring[2048];

      /*
       * Prepare for bailout possibilities by setting a signal handler for
       * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack
       * environment with sigsetjmp (for the signal handler) and setjmp
       * (for the "bail out" utility function).
       */
      if ( EF_Util_setsig("efcn_get_custom_axes")) {
          *status = FERR_EF_ERROR;
           return;
      }
      if (sigsetjmp(sigjumpbuffer, 1) != 0) {
          *status = FERR_EF_ERROR;
          return;
      }
      if (setjmp(jumpbuffer) != 0) {
          *status = FERR_EF_ERROR;
          return;
      }
      canjump = 1;

      /* Call pyefcn_custom_axes which in turn calls the ferret_custom_axes method in the python module */
      pyefcn_custom_axes(*id_ptr, ef_ptr->path, errstring);
      if ( strlen(errstring) > 0 ) {
          /* (In effect) call ef_bail_out_ to process the error in a standard way */
          FORTRAN(ef_err_bail_out)(id_ptr, errstring);
          /* Should never return - instead jumps to setjmp() returning 1 */
      }

      /* Restore the old signal handlers. */
      EF_Util_ressig("efcn_get_custom_axes");

      /* end of EF_PYTHON */
  }
  else {
    *status = FERR_EF_ERROR;
    fprintf(stderr, "**ERROR: unsupported language (%d) for efcn_get_custom_axes.\n", ef_ptr->internals_ptr->language);
  }

  return;
}


/*
 * Find an external function based on its integer ID,
 * Query the function about abstract axes. Pass memory,
 * mr_list and cx_list info into the external function.
 * 1/17 *SH* removed argument "memory" from the calling arguments
 *           It was never used by the routine, anyway.
 */
void FORTRAN(efcn_get_result_limits)( int *id_ptr, int *mr_list_ptr, int *cx_list_ptr, int *status )
{
  ExternalFunction *ef_ptr=NULL;
  char tempText[EF_MAX_NAME_LENGTH]="";
  int internally_linked = FALSE;

  void (*fptr)(int *);

  /*
   * Initialize the status
   */
  *status = FERR_OK;

  /*
   * Store the memory pointer and various lists globally.
   */
  EF_store_globals(mr_list_ptr, cx_list_ptr, NULL, NULL);

  /*
   * Find the external function.
   */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  if ( (!strcmp(ef_ptr->path,"internally_linked")) ) {internally_linked = TRUE; }

  if ( ef_ptr->internals_ptr->language == EF_F ) {

    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack
     * environment with sigsetjmp (for the signal handler) and setjmp
     * (for the "bail out" utility function).
     */


    if ( EF_Util_setsig("efcn_get_result_limits")) {
      *status = FERR_EF_ERROR;
       return;
    }

    /*
     * Set the signal return location and process jumps
     */
    if (sigsetjmp(sigjumpbuffer, 1) != 0) {
      *status = FERR_EF_ERROR;
      return;
    }

    /*
     * Set the bail out return location and process jumps
     */
    if (setjmp(jumpbuffer) != 0) {
      *status = FERR_EF_ERROR;
      return;
    }

    canjump = 1;


    strcpy(tempText, ef_ptr->name);
    strcat(tempText, "_result_limits_");

    if (!internally_linked) {
      fptr  = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
    } else {
      fptr  = (void (*)(int *))internal_dlsym(tempText);
    }

    (*fptr)( id_ptr);

    /*
     * Restore the old signal handlers.
     */
    if ( EF_Util_ressig("efcn_get_result_limits")) {
       return;
    }

    /* end of EF_F */
  }
  else if ( ef_ptr->internals_ptr->language == EF_PYTHON ) {
      char errstring[2048];

      /*
       * Prepare for bailout possibilities by setting a signal handler for
       * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack
       * environment with sigsetjmp (for the signal handler) and setjmp
       * (for the "bail out" utility function).
       */
      if ( EF_Util_setsig("efcn_get_result_limits")) {
          *status = FERR_EF_ERROR;
           return;
      }
      if (sigsetjmp(sigjumpbuffer, 1) != 0) {
          *status = FERR_EF_ERROR;
          return;
      }
      if (setjmp(jumpbuffer) != 0) {
          *status = FERR_EF_ERROR;
          return;
      }
      canjump = 1;

      /* Call pyefcn_result_limits which in turn calls the ferret_result_limits method in the python module */
      pyefcn_result_limits(*id_ptr, ef_ptr->path, errstring);
      if ( strlen(errstring) > 0 ) {
          /* (In effect) call ef_bail_out_ to process the error in a standard way */
          FORTRAN(ef_err_bail_out)(id_ptr, errstring);
          /* Should never return - instead jumps to setjmp() returning 1 */
      }

      /* Restore the old signal handlers. */
      EF_Util_ressig("efcn_get_result_limits");

      /* end of EF_PYTHON */
  }
  else {
    *status = FERR_EF_ERROR;
    fprintf(stderr, "**ERROR: unsupported language (%d) for efcn_get_result_limits.\n", ef_ptr->internals_ptr->language);
  }

  return;
}


/*
 * Find an external function based on its integer ID,
 * pass the necessary information and the data and tell
 * the function to calculate the result.
 */
void FORTRAN(efcn_compute)( int *id_ptr, int *narg_ptr, int *cx_list_ptr, int *mr_list_ptr, int *mres_ptr,
	DFTYPE *bad_flag_ptr, int *status )
{
  /* 
   * The array of work array memory pointers are used in return setjmp/longjmp 
   * and sigsetjmp/siglongjmp blocks so cannot be an normal automatic variable.
   */
  static DFTYPE *(work_ptr[EF_MAX_WORK_ARRAYS]);

  ExternalFunction *ef_ptr=NULL;
  ExternalFunctionInternals *i_ptr=NULL;
  DFTYPE *arg_ptr[EF_MAX_COMPUTE_ARGS];
  int i=0, j=0;
  int size=0;
  int nargs=0;
  char tempText[EF_MAX_NAME_LENGTH]="";
  int internally_linked = FALSE;

  /*
   * Pointers to all the functions (with protoypes) needed 
   * for varying numbers of arguments and work arrays.
   */

  void (*copy_ferret_ef_mem_subsc_ptr)(void);
  void (*fptr)(int *);
  void (*f1arg)(int *, DFTYPE *, DFTYPE *);
  void (*f2arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f3arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f4arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f5arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f6arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *);
  void (*f7arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *);
  void (*f8arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f9arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f10arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f11arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f12arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f13arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f14arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
        DFTYPE *);
  void (*f15arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
        DFTYPE *, DFTYPE *);
  void (*f16arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
        DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f17arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
        DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);
  void (*f18arg)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
		DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
        DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *);

  /*
   * Initialize the status
   */
  *status = FERR_OK;

  /*
   * Store the array dimensions for memory resident variables and for working storage.
   * Store the memory pointer and various lists globally.
   */
  FORTRAN(efcn_copy_array_dims)();
  EF_store_globals(mr_list_ptr, cx_list_ptr, mres_ptr, bad_flag_ptr);

  /*
   * Find the external function.
   */
  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) {
    fprintf(stderr, "**ERROR in efcn_compute() finding external function: id = [%d]\n", *id_ptr);
    *status = FERR_EF_ERROR;
    return;
  }
  if ( (!strcmp(ef_ptr->path,"internally_linked")) ) {internally_linked = TRUE; }

  i_ptr = ef_ptr->internals_ptr;

/*
   1/17 tell FORTRAN to pass the pointers (place them into GLOBALs)
*/
  nargs = i_ptr->num_reqd_args;
  FORTRAN(efcn_rqst_mr_ptrs)(&nargs, mr_list_ptr, mres_ptr);

  if ( i_ptr->language == EF_F ) {
    /*
     * Begin assigning the arg_ptrs.
     */


    /* First come the arguments to the function. */

     for (i=0; i<i_ptr->num_reqd_args; i++) {
       arg_ptr[i] = GLOBAL_arg_ptrs[i];
     }

    /* Now for the result */

     arg_ptr[i++] = GLOBAL_res_ptr;


    /*
	  fprintf(stdout, "Calling compute routine for external function %s\n",
                          ef_ptr->name, i_ptr->num_work_arrays, EF_MAX_WORK_ARRAYS);
 */

    /* Now for the work arrays */

    /*
     * If this program has requested working storage we need to
     * ask the function to specify the amount of space needed
     * and then create the memory here.  Memory will be released
     * after the external function returns.
     */
    if (i_ptr->num_work_arrays > EF_MAX_WORK_ARRAYS) {

	  fprintf(stderr, "**ERROR specifying number of work arrays in ~_init subroutine of external function %s\n"
                          "\tnum_work_arrays[=%d] exceeds maximum[=%d].\n\n",
                          ef_ptr->name, i_ptr->num_work_arrays, EF_MAX_WORK_ARRAYS);
	  *status = FERR_EF_ERROR;
	  return;

    } else if (i_ptr->num_work_arrays < 0) {

	  fprintf(stderr, "**ERROR specifying number of work arrays in ~_init subroutine of external function %s\n"
                          "\tnum_work_arrays[=%d] must be a positive number.\n\n",
                          ef_ptr->name, i_ptr->num_work_arrays);
	  *status = FERR_EF_ERROR;
	  return;

    } else if (i_ptr->num_work_arrays > 0)  {

      strcpy(tempText, ef_ptr->name);
      strcat(tempText, "_work_size_");

      if (!internally_linked) {
         fptr = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
      } else {
         fptr  = (void (*)(int *))internal_dlsym(tempText);
      }

      if (fptr == NULL) {
	fprintf(stderr, "**ERROR in efcn_compute() accessing %s\n", tempText);
	*status = FERR_EF_ERROR;
        return;
      }
      (*fptr)( id_ptr );


      /* Allocate memory for each individual work array */
      for (j = 0; j < EF_MAX_WORK_ARRAYS; j++)
         work_ptr[j] = NULL;
      for (j=0; j<i_ptr->num_work_arrays; i++, j++) {

        int iarray, xlo, ylo, zlo, tlo, elo, flo,
                    xhi, yhi, zhi, thi, ehi, fhi;
        iarray = j+1;
        xlo = i_ptr->work_array_lo[j][0];
        ylo = i_ptr->work_array_lo[j][1];
        zlo = i_ptr->work_array_lo[j][2];
        tlo = i_ptr->work_array_lo[j][3];
        elo = i_ptr->work_array_lo[j][4];
        flo = i_ptr->work_array_lo[j][5];
        xhi = i_ptr->work_array_hi[j][0];
        yhi = i_ptr->work_array_hi[j][1];
        zhi = i_ptr->work_array_hi[j][2];
        thi = i_ptr->work_array_hi[j][3];
        ehi = i_ptr->work_array_hi[j][4];
        fhi = i_ptr->work_array_hi[j][5];

        FORTRAN(efcn_set_work_array_dims)(&iarray, &xlo, &ylo, &zlo, &tlo, &elo, &flo,
                                                   &xhi, &yhi, &zhi, &thi, &ehi, &fhi);

        size = sizeof(DFTYPE) * (xhi-xlo+1) * (yhi-ylo+1) * (zhi-zlo+1)
                              * (thi-tlo+1) * (ehi-elo+1) * (fhi-flo+1);

        arg_ptr[i] = (DFTYPE *)FerMem_Malloc(size, __FILE__, __LINE__);
        if ( arg_ptr[i] == NULL ) {
          fprintf(stderr, "**ERROR in efcn_compute() allocating %d bytes of memory\n"
                          "\twork array %d:  X=%d:%d, Y=%d:%d, Z=%d:%d, T=%d:%d, E=%d:%d, F=%d:%d\n",
                          size, iarray, xlo, xhi, ylo, yhi, zlo, zhi, tlo, thi, elo, ehi, flo, fhi);
          while ( j > 0 ) {
             j--;
             FerMem_Free(work_ptr[j], __FILE__, __LINE__);
             work_ptr[j] = NULL;
          }
	  *status = FERR_EF_ERROR;
	  return;
        }
        work_ptr[j] = arg_ptr[i];
      }

    }

    if ( ! internally_linked ) {
        /*
         * Copy the memory subscripts to the copy of the EF_MEM_SUBSC 
         * common block found in the external function.  The EF_MEM_SUBSC
         * common block in Ferret is not visible to the external function
         * because the pyferret module is a shared object library that
         * Python has loaded privately.
         */
        copy_ferret_ef_mem_subsc_ptr = (void (*)(void)) 
                dlsym(ef_ptr->handle, "copy_ferret_ef_mem_subsc_");
        if ( copy_ferret_ef_mem_subsc_ptr == NULL ) {
            fprintf(stderr, "**ERROR: efcn_scan: copy_ferret_ef_mem_subsc_\n"
                            "  not found -- %s\n", dlerror());
            *status = FERR_EF_ERROR;
            return;
        }
        (*copy_ferret_ef_mem_subsc_ptr)();
    }

    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack
     * environment with sigsetjmp (for the signal handler) and setjmp
     * (for the "bail out" utility function).
     */

    if ( EF_Util_setsig("efcn_compute")) {
      for (j = 0; j < EF_MAX_WORK_ARRAYS; j++) {
        if ( work_ptr[j] == NULL )
          break;
        FerMem_Free(work_ptr[j], __FILE__, __LINE__);
        work_ptr[j] = NULL;
      }
      *status = FERR_EF_ERROR;
      return;
    }

    /*
     * Set the signal return location and process jumps
     */
    if (sigsetjmp(sigjumpbuffer, 1) != 0) {
      for (j = 0; j < EF_MAX_WORK_ARRAYS; j++) {
        if ( work_ptr[j] == NULL )
          break;
        FerMem_Free(work_ptr[j], __FILE__, __LINE__);
        work_ptr[j] = NULL;
      }
      *status = FERR_EF_ERROR;
      return;
    }

    /*
     * Set the bail out return location and process jumps
     */
    if (setjmp(jumpbuffer) != 0) {
      for (j = 0; j < EF_MAX_WORK_ARRAYS; j++) {
        if ( work_ptr[j] == NULL )
          break;
        FerMem_Free(work_ptr[j], __FILE__, __LINE__);
        work_ptr[j] = NULL;
      }
      *status = FERR_EF_ERROR;
      return;
    }

    canjump = 1;

    /*
     * Now go ahead and call the external function's "_compute_" function,
     * prototyping it for the number of arguments expected.
     */
    strcpy(tempText, ef_ptr->name);
    strcat(tempText, "_compute_");

    switch ( i_ptr->num_reqd_args + i_ptr->num_work_arrays ) {

    case 1:
	  if (!internally_linked) {
            f1arg  = (void (*)(int *, DFTYPE *, DFTYPE *))
             dlsym(ef_ptr->handle, tempText);
          } else {
	    f1arg  = (void (*)(int *, DFTYPE *, DFTYPE *))
             internal_dlsym(tempText);
          }
	  (*f1arg)( id_ptr, arg_ptr[0], arg_ptr[1] );
	break;


    case 2:
	  if (!internally_linked) {
            f2arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *))
             dlsym(ef_ptr->handle, tempText);
          } else {
            f2arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *))
             internal_dlsym(tempText);
          }
	  (*f2arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2] );
	break;


    case 3:
	  if (!internally_linked) {
	     f3arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
              dlsym(ef_ptr->handle, tempText);
          } else {
	     f3arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
              internal_dlsym(tempText);
          }
	  (*f3arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3] );
	break;


    case 4:
	  if (!internally_linked) {
            f4arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
             dlsym(ef_ptr->handle, tempText);
          } else {
            f4arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
             internal_dlsym(tempText);
          }
	  (*f4arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4] );
	break;


    case 5:
	  if (!internally_linked) {
	    f5arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *))
             dlsym(ef_ptr->handle, tempText);
          } else {
	    f5arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *))
             internal_dlsym(tempText);
          }
	  (*f5arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5] );
	break;


    case 6:
	  if (!internally_linked) {
	    f6arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f6arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *))internal_dlsym(tempText);
          }
	  (*f6arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6] );
	break;


    case 7:
	  if (!internally_linked) {
	    f7arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f7arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *))internal_dlsym(tempText);
          }
	  (*f7arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7] );
	break;


    case 8:
	  if (!internally_linked) {
	    f8arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f8arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *))internal_dlsym(tempText);
          }
	  (*f8arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8] );
	break;


    case 9:
	  if (!internally_linked) {
            f9arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
            f9arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))internal_dlsym(tempText);
          }
	  (*f9arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9] );
	break;


    case 10:
	  if (!internally_linked) {
	    f10arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f10arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))internal_dlsym(tempText);
          }
	  (*f10arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10] );
	break;


    case 11:
	  if (!internally_linked) {
            f11arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
             dlsym(ef_ptr->handle, tempText);
          } else {
            f11arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
             internal_dlsym(tempText);
          }
	  (*f11arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11] );
	break;


    case 12:
	  if (!internally_linked) {
	    f12arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
             dlsym(ef_ptr->handle, tempText);
          } else {
	    f12arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
             internal_dlsym(tempText);
          }
	  (*f12arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12] );
	break;


    case 13:
	  if (!internally_linked) {
	    f13arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
             dlsym(ef_ptr->handle, tempText);
          } else {
	    f13arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))
             internal_dlsym(tempText);
          }
	  (*f13arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13] );
	break;


    case 14:
	  if (!internally_linked) {
	    f14arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f14arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *))internal_dlsym(tempText);
          }
	  (*f14arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14] );
	break;


    case 15:
	  if (!internally_linked) {
	   f15arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
            DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
            DFTYPE *, DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
	   f15arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
            DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
            DFTYPE *, DFTYPE *))internal_dlsym(tempText);
          }
	  (*f15arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15] );
	break;


    case 16:
	  if (!internally_linked) {
	    f16arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f16arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *))internal_dlsym(tempText);
          }
	  (*f16arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15], arg_ptr[16] );
	break;


    case 17:
	  if (!internally_linked) {
            f17arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
            f17arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))internal_dlsym(tempText);
          }
	  (*f17arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15], arg_ptr[16],
        arg_ptr[17] );
	break;


    case 18:
	  if (!internally_linked) {
	    f18arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f18arg  = (void (*)(int *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *,
             DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *, DFTYPE *))internal_dlsym(tempText);
          }
	  (*f18arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15], arg_ptr[16],
        arg_ptr[17], arg_ptr[18] );
	break;


    default:
      for (j = 0; j < EF_MAX_WORK_ARRAYS; j++) {
        if ( work_ptr[j] == NULL )
          break;
        FerMem_Free(work_ptr[j], __FILE__, __LINE__);
        work_ptr[j] = NULL;
      }
      fprintf(stderr, "**ERROR: External functions with more than %d arguments are not implemented.\n",
                      EF_MAX_ARGS);
      *status = FERR_EF_ERROR;
      return;
      break;

    }

    /* Release the work space. */
    for (j = 0; j < EF_MAX_WORK_ARRAYS; j++) {
      if ( work_ptr[j] == NULL )
        break;
      FerMem_Free(work_ptr[j], __FILE__, __LINE__);
      work_ptr[j] = NULL;
    }

    /*
     * Restore the old signal handlers.
     */
    if ( EF_Util_ressig("efcn_compute")) {
      *status = FERR_EF_ERROR;
      return;
    }

    /* Success for EF_F */
  }
  else if ( i_ptr->language == EF_PYTHON ) {
      int   memlo[EF_MAX_COMPUTE_ARGS][NFERDIMS], memhi[EF_MAX_COMPUTE_ARGS][NFERDIMS],
            steplo[EF_MAX_COMPUTE_ARGS][NFERDIMS], stephi[EF_MAX_COMPUTE_ARGS][NFERDIMS],
            incr[EF_MAX_COMPUTE_ARGS][NFERDIMS];
      DFTYPE badflags[EF_MAX_COMPUTE_ARGS];
      char  errstring[2048];


      /* First the results grid array, then the argument grid arrays */
      arg_ptr[0] = GLOBAL_res_ptr; // 1/17 *sh*
      for (i = 0; i < i_ptr->num_reqd_args; i++) {
          arg_ptr[i+1] = GLOBAL_arg_ptrs[i];
      }

      /* Assign the memory limits, step values, and bad-data-flag values - first result, then arguments */
      FORTRAN(ef_get_res_mem_subscripts_6d)(id_ptr, memlo[0], memhi[0]);
      FORTRAN(ef_get_arg_mem_subscripts_6d)(id_ptr, &(memlo[1]), &(memhi[1]));
      FORTRAN(ef_get_res_subscripts_6d)(id_ptr, steplo[0], stephi[0], incr[0]);
      FORTRAN(ef_get_arg_subscripts_6d)(id_ptr, &(steplo[1]), &(stephi[1]), &(incr[1]));
      FORTRAN(ef_get_bad_flags)(id_ptr, &(badflags[1]), &(badflags[0]));

      /* Reset zero increments to +1 or -1 for pyefcn_compute */
      for (i = 0; i <= i_ptr->num_reqd_args; i++) {
          for (j = 0; j < NFERDIMS; j++) {
              if ( incr[i][j] == 0 ) {
                  if ( steplo[i][j] <= stephi[i][j] )
                      incr[i][j] = 1;
                  else
                      incr[i][j] = -1;
              }
          }
      }

      /*
       * Prepare for bailout possibilities by setting a signal handler for
       * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack
       * environment with sigsetjmp (for the signal handler) and setjmp
       * (for the "bail out" utility function).
       */
      if ( EF_Util_setsig("efcn_compute")) {
          *status = FERR_EF_ERROR;
          return;
      }
      if (sigsetjmp(sigjumpbuffer, 1) != 0) {
          *status = FERR_EF_ERROR;
          return;
      }
      if (setjmp(jumpbuffer) != 0) {
          *status = FERR_EF_ERROR;
          return;
      }
      canjump = 1;

      /* Call pyefcn_compute which in turn calls the ferret_compute method in the python module */
      pyefcn_compute(*id_ptr, ef_ptr->path, arg_ptr, (i_ptr->num_reqd_args)+1, memlo, memhi, steplo, stephi, incr, badflags, errstring);
      if ( strlen(errstring) > 0 ) {
          /* (In effect) call ef_bail_out_ to process the error in a standard way */
          FORTRAN(ef_err_bail_out)(id_ptr, errstring);
          /* Should never return - instead jumps to setjmp() returning 1 */
      }

      /* Restore the original signal handlers */
      EF_Util_ressig("efcn_compute");

      /* Success for EF_PYTHON */
  }
  else {
    fprintf(stderr, "**ERROR: unsupported language (%d) for efcn_compute.\n", i_ptr->language);
    *status = FERR_EF_ERROR;
  }

  return;
}


/*
 * A signal handler for SIGFPE, SIGSEGV, SIGINT and SIGBUS signals generated
 * while executing an external function.  See "Advanced Programming
 * in the UNIX Environment" p. 299 ff for details.
 *
 * This routine should never return since a signal was raised indicating a
 * problem.  The siglongjump rewinds back to where sigsetjmp was called with
 * the current sigjumpbuffer.
 */
static void EF_signal_handler(int signo)
{
   if ( canjump == 0 ) {
      fprintf(stderr, "EF_signal_handler invoked with signal %d but canjump = 0", signo);
      fflush(stderr);
      abort();
   }

   /*
    * Restore the old signal handlers.
    */
   if ( EF_Util_ressig("efcn_compute")) {
      /* error message already printed */
      fflush(stderr);
      abort();
   }

   if (signo == SIGFPE) {
      fprintf(stderr, "**ERROR in external function: Floating Point Error\n");
      canjump = 0;
      siglongjmp(sigjumpbuffer, 1);
   } else if (signo == SIGSEGV) {
      fprintf(stderr, "**ERROR in external function: Segmentation Violation\n");
      canjump = 0;
      siglongjmp(sigjumpbuffer, 1);
   } else if (signo == SIGINT) {
      fprintf(stderr, "**External function halted with Control-C\n");
      canjump = 0;
      siglongjmp(sigjumpbuffer, 1);
   } else if (signo == SIGBUS) {
      fprintf(stderr, "**ERROR in external function: Hardware Fault\n");
      canjump = 0;
      siglongjmp(sigjumpbuffer, 1);
   } else {
      fprintf(stderr, "**ERROR in external function: signo = %d\n", signo);
      canjump = 0;
      siglongjmp(sigjumpbuffer, 1);
   }

}


/*
 * Find an external function based on its name and
 * return the integer ID associated with that funciton.
 */
int FORTRAN(efcn_get_id)( char name[] )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;

  static int return_val=0; /* static because it needs to exist after the return statement */

  /*
   * Find the external function.
   */

  status = list_traverse(STATIC_ExternalFunctionList, name, EF_ListTraverse_FoundName,
                         (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, set the id_ptr to ATOM_NOT_FOUND.
   */
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }

  ef_ptr=(ExternalFunction *)list_curr(STATIC_ExternalFunctionList);

  return_val = ef_ptr->id;

  return return_val;
}


/*
 * Determine whether a function name matches a template.
 * Return 1 if the name matchs.
 */
int FORTRAN(efcn_match_template)( int *id_ptr, char template[] )
{
  ExternalFunction *ef_ptr=NULL;
  int EF_LT_MT_return;

  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return return_val; }

  EF_LT_MT_return = EF_ListTraverse_MatchTemplate((char *)template, (char *)ef_ptr);

  /* The list package forces 'list traversal' functions to return
   * 0 whenever a match is found.  We want to return a more reasonable
   * 1 (=true) if we find a match.
   */
  if ( EF_LT_MT_return == FALSE ) {
	return_val = 1;
  } else {
    return_val = 0;
  }

  return return_val;
}


/*
 */
void FORTRAN(efcn_get_custom_axis_sub)( int *id_ptr, int *axis_ptr, double *lo_ptr, double *hi_ptr,
			       double *del_ptr, char *unit, int *modulo_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  /*
   * Find the external function.
   */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(unit, ef_ptr->internals_ptr->axis[*axis_ptr-1].unit);
  *lo_ptr = ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_lo;
  *hi_ptr = ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_hi;
  *del_ptr = ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_del;
  *modulo_ptr = ef_ptr->internals_ptr->axis[*axis_ptr-1].modulo;

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the name.
 */
void FORTRAN(efcn_get_name)( int *id_ptr, char *name )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(name, ef_ptr->name);

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the version number.
 */
void FORTRAN(efcn_get_version)( int *id_ptr, DFTYPE *version )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  *version = ef_ptr->internals_ptr->version;

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the description.
 */
void FORTRAN(efcn_get_descr)( int *id_ptr, char *descr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(descr, ef_ptr->internals_ptr->description);

  return;
}

/*
 * Find an external function based on its integer ID and
 * return the name of an alternate function that operates
 * with string arguments.
 *
 * *kms* 2/11 - assign blank-terminated (not null-terminated)
 * string since code using this name expects this style.
 * Assumes alt_str_name has been intialized to all-blank.
 */
void FORTRAN(efcn_get_alt_type_fcn)( int *id_ptr, char *alt_str_name )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(alt_str_name, ef_ptr->internals_ptr->alt_fcn_name);
  alt_str_name[strlen(alt_str_name)] = ' ';

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the number of arguments.
 */
int FORTRAN(efcn_get_num_reqd_args)( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return return_val; }

  return_val = ef_ptr->internals_ptr->num_reqd_args;

  return return_val;
}


/*
 * Find an external function based on its integer ID and
 * return the flag stating whether the function has
 * a variable number of arguments.
 */
void FORTRAN(efcn_get_has_vari_args)( int *id_ptr, int *has_vari_args_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  *has_vari_args_ptr = ef_ptr->internals_ptr->has_vari_args;

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the axis sources (merged, normal, abstract, custom).
 */
void FORTRAN(efcn_get_axis_will_be)( int *id_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_will_be[X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_will_be[Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_will_be[Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_will_be[T_AXIS];
  array_ptr[E_AXIS] = ef_ptr->internals_ptr->axis_will_be[E_AXIS];
  array_ptr[F_AXIS] = ef_ptr->internals_ptr->axis_will_be[F_AXIS];

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the axis_reduction (retained, reduced) information.
 */
void FORTRAN(efcn_get_axis_reduction)( int *id_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_reduction[X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_reduction[Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_reduction[Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_reduction[T_AXIS];
  array_ptr[E_AXIS] = ef_ptr->internals_ptr->axis_reduction[E_AXIS];
  array_ptr[F_AXIS] = ef_ptr->internals_ptr->axis_reduction[F_AXIS];

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the piecemeal_ok information.  This lets Ferret
 * know if it's ok to break up a calculation along an axis
 * for memory management reasons.
 */
void FORTRAN(efcn_get_piecemeal_ok)( int *id_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[T_AXIS];
  array_ptr[E_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[E_AXIS];
  array_ptr[F_AXIS] = ef_ptr->internals_ptr->piecemeal_ok[F_AXIS];

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the (boolean) 'axis_implied_from' information for
 * a particular argument to find out if its axes should
 * be merged in to the result grid.
 */
void FORTRAN(efcn_get_axis_implied_from)( int *id_ptr, int *iarg_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][T_AXIS];
  array_ptr[E_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][E_AXIS];
  array_ptr[F_AXIS] = ef_ptr->internals_ptr->axis_implied_from[index][F_AXIS];

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the 'arg_extend_lo' information for a particular
 * argument which tells Ferret how much to extend axis limits
 * when providing input data (e.g. to compute a derivative).
 */
void FORTRAN(efcn_get_axis_extend_lo)( int *id_ptr, int *iarg_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][T_AXIS];
  array_ptr[E_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][E_AXIS];
  array_ptr[F_AXIS] = ef_ptr->internals_ptr->axis_extend_lo[index][F_AXIS];

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the 'arg_extend_hi' information for a particular
 * argument which tells Ferret how much to extend axis limits
 * when providing input data (e.g. to compute a derivative).
 */
void FORTRAN(efcn_get_axis_extend_hi)( int *id_ptr, int *iarg_ptr, int *array_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  array_ptr[X_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][X_AXIS];
  array_ptr[Y_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][Y_AXIS];
  array_ptr[Z_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][Z_AXIS];
  array_ptr[T_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][T_AXIS];
  array_ptr[E_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][E_AXIS];
  array_ptr[F_AXIS] = ef_ptr->internals_ptr->axis_extend_hi[index][F_AXIS];

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the 'axis_limits' information for a particular
 * argument.
 */
void FORTRAN(efcn_get_axis_limits)( int *id_ptr, int *axis_ptr, int *lo_ptr, int *hi_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *axis_ptr - 1; /* C indices are 1 less than Fortran */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  *lo_ptr = ef_ptr->internals_ptr->axis[index].ss_lo;
  *hi_ptr = ef_ptr->internals_ptr->axis[index].ss_hi;

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the 'arg_type' information for a particular
 * argument which tells Ferret whether an argument is a
 * DFTYPE or a string.
 */
int FORTRAN(efcn_get_arg_type)( int *id_ptr, int *iarg_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int return_val=0;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return return_val; }

  return_val = ef_ptr->internals_ptr->arg_type[index];

  return return_val;
}


/*
 * Find an external function based on its integer ID and
 * return the 'rtn_type' information for the result which
 * tells Ferret whether an argument is a DFTYPE or a string.
 */
int FORTRAN(efcn_get_rtn_type)( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return return_val; }

  return_val = ef_ptr->internals_ptr->return_type;

  return return_val;
}


/*
 * Find an external function based on its integer ID and
 * return the name of a particular argument.
 */
void FORTRAN(efcn_get_arg_name)( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */
  int i=0, printable=FALSE;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  /*
   * JC_NOTE: if the argument has no name then memory gets overwritten, corrupting
   * the address of iarg_ptr and causing a core dump.  I need to catch that case
   * here.
   */

  for (i=0;i<strlen(ef_ptr->internals_ptr->arg_name[index]);i++) {
    if (isgraph(ef_ptr->internals_ptr->arg_name[index][i])) {
      printable = TRUE;
      break;
    }
  }

  if ( printable ) {
    strcpy(string, ef_ptr->internals_ptr->arg_name[index]);
  } else {
    strcpy(string, "--");
  }

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the units for a particular argument.
 */
void FORTRAN(efcn_get_arg_unit)( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  ef_ptr=(ExternalFunction *)list_curr(STATIC_ExternalFunctionList);

  strcpy(string, ef_ptr->internals_ptr->arg_unit[index]);

  return;
}


/*
 * Find an external function based on its integer ID and
 * return the description of a particular argument.
 */
void FORTRAN(efcn_get_arg_desc)( int *id_ptr, int *iarg_ptr, char *string )
{
  ExternalFunction *ef_ptr=NULL;
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  strcpy(string, ef_ptr->internals_ptr->arg_desc[index]);

  return;
}



/*
 * This function should never return since there was a user-detected
 * error of some sort.  The call to longjump rewinds back to where
 * setjmp was called with the current jumpbuffer.
 */
void FORTRAN(ef_err_bail_out)(int *id_ptr, char *text)
{
   ExternalFunction *ef_ptr=NULL;

   ef_ptr = ef_ptr_from_id_ptr(id_ptr);
   if ( ef_ptr == NULL ) {
      fprintf(stderr, "Unknown external function ID of %d in ef_err_bail_out", *id_ptr);
      fflush(stderr);
      abort();
   }
   if ( canjump == 0 ) {
      fputs("ef_err_bail_out called with canjump = 0", stderr);
      fflush(stderr);
      abort();
   }
   /*
    * Restore the old signal handlers.
    */
   if ( EF_Util_ressig("efcn_compute")) {
      /* error message already printed */
      fflush(stderr);
      abort();
   }

   fprintf(stderr, "\n"
                   "Bailing out of external function \"%s\":\n"
                   "\t%s\n", ef_ptr->name, text);

   longjmp(jumpbuffer, 1);
}



/* .... Object Oriented Utility Functions .... */


/*
 * Allocate space for and initialize the internal
 * information for an EF.
 *
 * Return values:
 *     -1: error allocating space
 *      0: success
 */
int EF_New( ExternalFunction *this )
{
  ExternalFunctionInternals *i_ptr=NULL;
  int i=0, j=0;

  static int return_val=0; /* static because it needs to exist after the return statement */


  /*
   * Allocate space for the internals.
   * If the allocation failed, print a warning message and return.
   */

  this->internals_ptr = FerMem_Malloc(sizeof(ExternalFunctionInternals), __FILE__, __LINE__);
  i_ptr = this->internals_ptr;

  if ( i_ptr == NULL ) {
    fprintf(stderr, "**ERROR in EF_New(): cannot allocate ExternalFunctionInternals.\n");
    return_val = -1;
    return return_val;
  }


  /*
   * Initialize the internals.
   */

  /* Information about the overall function */

  i_ptr->version = EF_VERSION;
  strcpy(i_ptr->description, "");
  i_ptr->language = EF_F;
  i_ptr->num_reqd_args = 1;
  i_ptr->has_vari_args = NO;
  i_ptr->num_work_arrays = 0;
  i_ptr->return_type = FLOAT_RETURN;
  for (i=0; i<NFERDIMS; i++) {
    for (j=0; j<EF_MAX_WORK_ARRAYS; j++) {
      i_ptr->work_array_lo[j][i] = 1;
      i_ptr->work_array_hi[j][i] = 1;
    }
    i_ptr->axis_will_be[i] = IMPLIED_BY_ARGS;
    i_ptr->axis_reduction[i] = RETAINED;
    i_ptr->piecemeal_ok[i] = NO;
  }

  /* Information specific to each argument of the function */

  for (i=0; i<EF_MAX_ARGS; i++) {
    for (j=0; j<NFERDIMS; j++) {
      i_ptr->axis_implied_from[i][j] = YES;
      i_ptr->axis_extend_lo[i][j] = 0;
      i_ptr->axis_extend_hi[i][j] = 0;
    }
    i_ptr->arg_type[i] = FLOAT_ARG;
    strcpy(i_ptr->arg_name[i], "");
    strcpy(i_ptr->arg_unit[i], "");
    strcpy(i_ptr->arg_desc[i], "");
  }

  return return_val;

}


/* .... UtilityFunctions for dealing with STATIC_ExternalFunctionList .... */

/*
 * Store the global values which will be needed by utility routines
 * in EF_ExternalUtil.c
 */
void EF_store_globals(int *mr_list_ptr, int *cx_list_ptr,
	int *mres_ptr, DFTYPE *bad_flag_ptr)
{
  GLOBAL_mr_list_ptr = mr_list_ptr;
  GLOBAL_cx_list_ptr = cx_list_ptr;
  GLOBAL_mres_ptr = mres_ptr;
  GLOBAL_bad_flag_ptr = bad_flag_ptr;
}
void FORTRAN(efcn_pass_arg_ptr)(int *iarg, DFTYPE *arg_ptr)
{
  int iarg_c = *iarg-1;   // FORTRAN index to c index

  GLOBAL_arg_ptrs[iarg_c] = arg_ptr;
}

void FORTRAN(efcn_pass_res_ptr)(DFTYPE *res_ptr)
{
  GLOBAL_res_ptr = res_ptr;
}


/*
 * Find an external function based on an integer id
 * and return the pointer to the function.
 * Returns NULL if it fails.
 */
ExternalFunction *ef_ptr_from_id_ptr(int *id_ptr)
{
   ExternalFunction *ef_ptr;
   int status;

   /* Check if the list has been created to avoid a seg fault if called indiscriminately */
   if ( STATIC_ExternalFunctionList == NULL ) {
      return NULL;
   }

   /* Search the list for the function ID */
   status = list_traverse(STATIC_ExternalFunctionList, (char *) id_ptr, EF_ListTraverse_FoundID,
                          (LIST_FRNT | LIST_FORW | LIST_ALTR));
   if ( status != LIST_OK ) {
      return NULL;
   }

   /* Get the pointer to the function from the list */
   ef_ptr = (ExternalFunction *) list_curr(STATIC_ExternalFunctionList);
   return ef_ptr;
}


int EF_ListTraverse_fprintf( char *data, char *curr )
{
   ExternalFunction *ef_ptr=(ExternalFunction *)curr;

   fprintf(stderr, "path = \"%s\", name = \"%s\", id = %d, internals_ptr = %ld\n",
	   ef_ptr->path, ef_ptr->name, ef_ptr->id, (long) (ef_ptr->internals_ptr));

   return TRUE;
}


/*
 * Ferret always capitalizes everything so we'd better
 * be case INsensitive.
 */
int EF_ListTraverse_FoundName( char *data, char *curr )
{
  ExternalFunction *ef_ptr=(ExternalFunction *)curr;

  if ( !strcasecmp(data, ef_ptr->name) ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}


int EF_ListTraverse_MatchTemplate( char data[], char *curr )
{
  ExternalFunction *ef_ptr=(ExternalFunction *)curr;

  int i=0, star_skip=FALSE;
  char upname[EF_MAX_DESCRIPTION_LENGTH];
  char *t, *n;

  for (i=0; i<strlen(ef_ptr->name); i++) {
    upname[i] = toupper(ef_ptr->name[i]);
  }
  upname[i] = '\0';

  n = upname;

  for (i=0, t=data; i<strlen(data); i++, t++) {

    if ( *t == '*' ) {

      star_skip = TRUE;
      continue;

    } else if ( *t == '?' ) {

      if ( star_skip ) {
	continue;
      } else {
        n++;
	if ( *n == '\0' ) /* end of name */
	  return TRUE; /* no match */
	else
	  continue;
      }

    } else if ( star_skip ) {

      if ( (n = strchr(n, *t)) == NULL ) { /* character not found in rest of name */
	return TRUE; /* no match */
      } else {
	star_skip = FALSE;
      }

    } else if ( *n == '\0' ) /* end of name */
      return TRUE; /* no match */

    else if ( *t == *n ) {
      n++;
      continue;
    }

    else
      return TRUE; /* no match */

  }

  /* *sh* if any non-wildcard characters remain in the "curr" name, then reject
     probably a bug remains for a regexp ending in "?" */
  if ( *n == '\0' || star_skip )
    return FALSE; /* got all the way through: a match */
  else
    return TRUE; /* characters remain--e.g. "xx5" does not math regexp "xx" */

}


int EF_ListTraverse_FoundID( char *data, char *curr )
{
  ExternalFunction *ef_ptr=(ExternalFunction *)curr;
  int ID=*((int *)data);

  if ( ID == ef_ptr->id ) {
    return FALSE; /* found match */
  } else
    return TRUE;
}


int EF_Util_setsig(char fcn_name[])
{
    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack
     * environment with sigsetjmp (for the signal handler) and setjmp
     * (for the "bail out" utility function).
     */

    if ( (fpe_handler = signal(SIGFPE, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "**ERROR in %s() catching SIGFPE.\n", fcn_name);
      return 1;
    }
    if ( (segv_handler = signal(SIGSEGV, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "**ERROR in %s() catching SIGSEGV.\n", fcn_name);
      return 1;
    }
    if ( (int_handler = signal(SIGINT, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "**ERROR in %s() catching SIGINT.\n", fcn_name);
      return 1;
    }
    if ( (bus_handler = signal(SIGBUS, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "**ERROR in %s() catching SIGBUS.\n", fcn_name);
      return 1;
    }

    /* the setjmp and sigsetjmp code moved to in-line 10/00 --
     * longjump returns cannot be made reliably into a subroutine that may
     *no longer be active on the stack
     */

    return 0;
}


int EF_Util_ressig(char fcn_name[])
{
    /*
     * Restore the old signal handlers.
     */
    if (signal(SIGFPE, (*fpe_handler)) == SIG_ERR) {
      fprintf(stderr, "**ERROR in %s() restoring default SIGFPE handler.\n", fcn_name);
      return 1;
    }
    if (signal(SIGSEGV, (*segv_handler)) == SIG_ERR) {
      fprintf(stderr, "**ERROR in %s() restoring default SIGSEGV handler.\n", fcn_name);
      return 1;
    }
    if (signal(SIGINT, (*int_handler)) == SIG_ERR) {
      fprintf(stderr, "**ERROR in %s() restoring default SIGINT handler.\n", fcn_name);
      return 1;
    }
    if (signal(SIGBUS, (*bus_handler)) == SIG_ERR) {
      fprintf(stderr, "**ERROR in %s() restoring default SIGBUS handler.\n", fcn_name);
      return 1;
    }
    return 0;
}


/*
 *  ------------------------------------

 *  internal_dlsym
 *  Accept a string and return the function pointer
 *
 *  The names of all subroutines of internally linked EF's
 *  generated by the perl script int_dlsym.pl.  Check the
 *  first if statement - change else if to if.
 *
 *   ACM 2-25-00 Solaris and OSF both have the trailing
 *   underscore for statically-linked routines. */

static void *internal_dlsym(char *name) {

/* dsg_fmask_str.F */
if ( !strcmp(name,"dsg_fmask_str_init_") ) return (void *)FORTRAN(dsg_fmask_str_init);
else if ( !strcmp(name,"dsg_fmask_str_compute_") ) return (void *)FORTRAN(dsg_fmask_str_compute);

/* ffta.F */
if ( !strcmp(name,"ffta_init_") ) return (void *)FORTRAN(ffta_init);
else if ( !strcmp(name,"ffta_custom_axes_") ) return (void *)FORTRAN(ffta_custom_axes);
else if ( !strcmp(name,"ffta_result_limits_") ) return (void *)FORTRAN(ffta_result_limits);
else if ( !strcmp(name,"ffta_work_size_") ) return (void *)FORTRAN(ffta_work_size);
else if ( !strcmp(name,"ffta_compute_") ) return (void *)FORTRAN(ffta_compute);

/* fftp.F */
else if ( !strcmp(name,"fftp_init_") ) return (void *)FORTRAN(fftp_init);
else if ( !strcmp(name,"fftp_custom_axes_") ) return (void *)FORTRAN(fftp_custom_axes);
else if ( !strcmp(name,"fftp_result_limits_") ) return (void *)FORTRAN(fftp_result_limits);
else if ( !strcmp(name,"fftp_work_size_") ) return (void *)FORTRAN(fftp_work_size);
else if ( !strcmp(name,"fftp_compute_") ) return (void *)FORTRAN(fftp_compute);

/* fft_im.F */
else if ( !strcmp(name,"fft_im_init_") ) return (void *)FORTRAN(fft_im_init);
else if ( !strcmp(name,"fft_im_custom_axes_") ) return (void *)FORTRAN(fft_im_custom_axes);
else if ( !strcmp(name,"fft_im_result_limits_") ) return (void *)FORTRAN(fft_im_result_limits);
else if ( !strcmp(name,"fft_im_work_size_") ) return (void *)FORTRAN(fft_im_work_size);
else if ( !strcmp(name,"fft_im_compute_") ) return (void *)FORTRAN(fft_im_compute);

/* fft_inverse.F */
else if ( !strcmp(name,"fft_inverse_init_") ) return (void *)FORTRAN(fft_inverse_init);
else if ( !strcmp(name,"fft_inverse_result_limits_") ) return (void *)FORTRAN(fft_inverse_result_limits);
else if ( !strcmp(name,"fft_inverse_work_size_") ) return (void *)FORTRAN(fft_inverse_work_size);
else if ( !strcmp(name,"fft_inverse_compute_") ) return (void *)FORTRAN(fft_inverse_compute);

/* fft_re.F */
else if ( !strcmp(name,"fft_re_init_") ) return (void *)FORTRAN(fft_re_init);
else if ( !strcmp(name,"fft_re_custom_axes_") ) return (void *)FORTRAN(fft_re_custom_axes);
else if ( !strcmp(name,"fft_re_result_limits_") ) return (void *)FORTRAN(fft_re_result_limits);
else if ( !strcmp(name,"fft_re_work_size_") ) return (void *)FORTRAN(fft_re_work_size);
else if ( !strcmp(name,"fft_re_compute_") ) return (void *)FORTRAN(fft_re_compute);

/* sampleij.F */
else if ( !strcmp(name,"sampleij_init_") ) return (void *)FORTRAN(sampleij_init);
else if ( !strcmp(name,"sampleij_result_limits_") ) return (void *)FORTRAN(sampleij_result_limits);
else if ( !strcmp(name,"sampleij_work_size_") ) return (void *)FORTRAN(sampleij_work_size);
else if ( !strcmp(name,"sampleij_compute_") ) return (void *)FORTRAN(sampleij_compute);


/* samplei_multi.F */
else if ( !strcmp(name,"samplei_multi_init_") ) return (void *)FORTRAN(samplei_multi_init);
else if ( !strcmp(name,"samplei_multi_compute_") ) return (void *)FORTRAN(samplei_multi_compute);

/* samplej_multi.F */
else if ( !strcmp(name,"samplej_multi_init_") ) return (void *)FORTRAN(samplej_multi_init);
else if ( !strcmp(name,"samplej_multi_compute_") ) return (void *)FORTRAN(samplej_multi_compute);

/* samplek_multi.F */
else if ( !strcmp(name,"samplek_multi_init_") ) return (void *)FORTRAN(samplek_multi_init);
else if ( !strcmp(name,"samplek_multi_compute_") ) return (void *)FORTRAN(samplek_multi_compute);

/* samplel_multi.F */
else if ( !strcmp(name,"samplel_multi_init_") ) return (void *)FORTRAN(samplel_multi_init);
else if ( !strcmp(name,"samplel_multi_compute_") ) return (void *)FORTRAN(samplel_multi_compute);

/* samplem_multi.F */
else if ( !strcmp(name,"samplem_multi_init_") ) return (void *)FORTRAN(samplem_multi_init);
else if ( !strcmp(name,"samplem_multi_compute_") ) return (void *)FORTRAN(samplem_multi_compute);

/* samplen_multi.F */
else if ( !strcmp(name,"samplen_multi_init_") ) return (void *)FORTRAN(samplen_multi_init);
else if ( !strcmp(name,"samplen_multi_compute_") ) return (void *)FORTRAN(samplen_multi_compute);

/* samplet_date.F */
else if ( !strcmp(name,"samplet_date_init_") ) return (void *)FORTRAN(samplet_date_init);
else if ( !strcmp(name,"samplet_date_result_limits_") ) return (void *)FORTRAN(samplet_date_result_limits);
else if ( !strcmp(name,"samplet_date_work_size_") ) return (void *)FORTRAN(samplet_date_work_size);
else if ( !strcmp(name,"samplet_date_compute_") ) return (void *)FORTRAN(samplet_date_compute);

/* samplef_date.F */
else if ( !strcmp(name,"samplef_date_init_") ) return (void *)FORTRAN(samplef_date_init);
else if ( !strcmp(name,"samplef_date_result_limits_") ) return (void *)FORTRAN(samplef_date_result_limits);
else if ( !strcmp(name,"samplef_date_work_size_") ) return (void *)FORTRAN(samplef_date_work_size);
else if ( !strcmp(name,"samplef_date_compute_") ) return (void *)FORTRAN(samplef_date_compute);

/* samplexy.F */
else if ( !strcmp(name,"samplexy_init_") ) return (void *)FORTRAN(samplexy_init);
else if ( !strcmp(name,"samplexy_result_limits_") ) return (void *)FORTRAN(samplexy_result_limits);
else if ( !strcmp(name,"samplexy_work_size_") ) return (void *)FORTRAN(samplexy_work_size);
else if ( !strcmp(name,"samplexy_compute_") ) return (void *)FORTRAN(samplexy_compute);

/* samplexyt.F */
else if ( !strcmp(name,"samplexyt_init_") ) return (void *)FORTRAN(samplexyt_init);
else if ( !strcmp(name,"samplexyt_result_limits_") ) return (void *)FORTRAN(samplexyt_result_limits);
else if ( !strcmp(name,"samplexyt_work_size_") ) return (void *)FORTRAN(samplexyt_work_size);
else if ( !strcmp(name,"samplexyt_compute_") ) return (void *)FORTRAN(samplexyt_compute);

/* samplexyz.F */
else if ( !strcmp(name,"samplexyz_init_") ) return (void *)FORTRAN(samplexyz_init);
else if ( !strcmp(name,"samplexyz_result_limits_") ) return (void *)FORTRAN(samplexyz_result_limits);
else if ( !strcmp(name,"samplexyz_work_size_") ) return (void *)FORTRAN(samplexyz_work_size);
else if ( !strcmp(name,"samplexyz_compute_") ) return (void *)FORTRAN(samplexyz_compute);

/* samplexyzt.F */
else if ( !strcmp(name,"samplexyzt_init_") ) return (void *)FORTRAN(samplexyzt_init);
else if ( !strcmp(name,"samplexyzt_result_limits_") ) return (void *)FORTRAN(samplexyzt_result_limits);
else if ( !strcmp(name,"samplexyzt_work_size_") ) return (void *)FORTRAN(samplexyzt_work_size);
else if ( !strcmp(name,"samplexyzt_compute_") ) return (void *)FORTRAN(samplexyzt_compute);

/* samplexyt_nrst.F */
else if ( !strcmp(name,"samplexyt_nrst_init_") ) return (void *)FORTRAN(samplexyt_nrst_init);
else if ( !strcmp(name,"samplexyt_nrst_result_limits_") ) return (void *)FORTRAN(samplexyt_nrst_result_limits);
else if ( !strcmp(name,"samplexyt_nrst_work_size_") ) return (void *)FORTRAN(samplexyt_nrst_work_size);
else if ( !strcmp(name,"samplexyt_nrst_compute_") ) return (void *)FORTRAN(samplexyt_nrst_compute);

/* samplexy_curv.F */
else if ( !strcmp(name,"samplexy_curv_init_") ) return (void *)FORTRAN(samplexy_curv_init);
else if ( !strcmp(name,"samplexy_curv_result_limits_") ) return (void *)FORTRAN(samplexy_curv_result_limits);
else if ( !strcmp(name,"samplexy_curv_work_size_") ) return (void *)FORTRAN(samplexy_curv_work_size);
else if ( !strcmp(name,"samplexy_curv_compute_") ) return (void *)FORTRAN(samplexy_curv_compute);

/* samplexy_curv_avg.F */
else if ( !strcmp(name,"samplexy_curv_avg_init_") ) return (void *)FORTRAN(samplexy_curv_avg_init);
else if ( !strcmp(name,"samplexy_curv_avg_result_limits_") ) return (void *)FORTRAN(samplexy_curv_avg_result_limits);
else if ( !strcmp(name,"samplexy_curv_avg_work_size_") ) return (void *)FORTRAN(samplexy_curv_avg_work_size);
else if ( !strcmp(name,"samplexy_curv_avg_compute_") ) return (void *)FORTRAN(samplexy_curv_avg_compute);

/* samplexy_curv_nrst.F */
else if ( !strcmp(name,"samplexy_curv_nrst_init_") ) return (void *)FORTRAN(samplexy_curv_nrst_init);
else if ( !strcmp(name,"samplexy_curv_nrst_result_limits_") ) return (void *)FORTRAN(samplexy_curv_nrst_result_limits);
else if ( !strcmp(name,"samplexy_curv_nrst_work_size_") ) return (void *)FORTRAN(samplexy_curv_nrst_work_size);
else if ( !strcmp(name,"samplexy_curv_nrst_compute_") ) return (void *)FORTRAN(samplexy_curv_nrst_compute);

/* samplexy_closest.F */
else if ( !strcmp(name,"samplexy_closest_init_") ) return (void *)FORTRAN(samplexy_closest_init);
else if ( !strcmp(name,"samplexy_closest_compute_") ) return (void *)FORTRAN(samplexy_closest_compute);

/* samplexy_nrst.F */
else if ( !strcmp(name,"samplexy_nrst_init_") ) return (void *)FORTRAN(samplexy_nrst_init);
else if ( !strcmp(name,"samplexy_nrst_result_limits_") ) return (void *)FORTRAN(samplexy_nrst_result_limits);
else if ( !strcmp(name,"samplexy_nrst_work_size_") ) return (void *)FORTRAN(samplexy_nrst_work_size);
else if ( !strcmp(name,"samplexy_nrst_compute_") ) return (void *)FORTRAN(samplexy_nrst_compute);

/* samplexz.F */
else if ( !strcmp(name,"samplexz_init_") ) return (void *)FORTRAN(samplexz_init);
else if ( !strcmp(name,"samplexz_result_limits_") ) return (void *)FORTRAN(samplexz_result_limits);
else if ( !strcmp(name,"samplexz_work_size_") ) return (void *)FORTRAN(samplexz_work_size);
else if ( !strcmp(name,"samplexz_compute_") ) return (void *)FORTRAN(samplexz_compute);

/* sampleyz.F */
else if ( !strcmp(name,"sampleyz_init_") ) return (void *)FORTRAN(sampleyz_init);
else if ( !strcmp(name,"sampleyz_result_limits_") ) return (void *)FORTRAN(sampleyz_result_limits);
else if ( !strcmp(name,"sampleyz_work_size_") ) return (void *)FORTRAN(sampleyz_work_size);
else if ( !strcmp(name,"sampleyz_compute_") ) return (void *)FORTRAN(sampleyz_compute);

/* scat2grid_bin_xy.F */
else if ( !strcmp(name,"scat2grid_bin_xy_init_") ) return (void *)FORTRAN(scat2grid_bin_xy_init);
else if ( !strcmp(name,"scat2grid_bin_xy_work_size_") ) return (void *)FORTRAN(scat2grid_bin_xy_work_size);
else if ( !strcmp(name,"scat2grid_bin_xy_compute_") ) return (void *)FORTRAN(scat2grid_bin_xy_compute);

/* scat2grid_bin_xyt.F */
else if ( !strcmp(name,"scat2grid_bin_xyt_init_") ) return (void *)FORTRAN(scat2grid_bin_xyt_init);
else if ( !strcmp(name,"scat2grid_bin_xyt_work_size_") ) return (void *)FORTRAN(scat2grid_bin_xyt_work_size);
else if ( !strcmp(name,"scat2grid_bin_xyt_compute_") ) return (void *)FORTRAN(scat2grid_bin_xyt_compute);

/* scat2grid_bin_xyz.F */
else if ( !strcmp(name,"scat2grid_bin_xyz_init_") ) return (void *)FORTRAN(scat2grid_bin_xyz_init);
else if ( !strcmp(name,"scat2grid_bin_xyz_work_size_") ) return (void *)FORTRAN(scat2grid_bin_xyz_work_size);
else if ( !strcmp(name,"scat2grid_bin_xyz_compute_") ) return (void *)FORTRAN(scat2grid_bin_xyz_compute);

/* scat2grid_bin_xyzt.F */
else if ( !strcmp(name,"scat2grid_bin_xyzt_init_") ) return (void *)FORTRAN(scat2grid_bin_xyzt_init);
else if ( !strcmp(name,"scat2grid_bin_xyzt_work_size_") ) return (void *)FORTRAN(scat2grid_bin_xyzt_work_size);
else if ( !strcmp(name,"scat2grid_bin_xyzt_compute_") ) return (void *)FORTRAN(scat2grid_bin_xyzt_compute);

/* scat2grid_nbin_xy.F */
else if ( !strcmp(name,"scat2grid_nbin_xy_init_") ) return (void *)FORTRAN(scat2grid_nbin_xy_init);
else if ( !strcmp(name,"scat2grid_nbin_xy_work_size_") ) return (void *)FORTRAN(scat2grid_nbin_xy_work_size);
else if ( !strcmp(name,"scat2grid_nbin_xy_compute_") ) return (void *)FORTRAN(scat2grid_nbin_xy_compute);

/* scat2grid_nbin_xyt.F */
else if ( !strcmp(name,"scat2grid_nbin_xyt_init_") ) return (void *)FORTRAN(scat2grid_nbin_xyt_init);
else if ( !strcmp(name,"scat2grid_nbin_xyt_work_size_") ) return (void *)FORTRAN(scat2grid_nbin_xyt_work_size);
else if ( !strcmp(name,"scat2grid_nbin_xyt_compute_") ) return (void *)FORTRAN(scat2grid_nbin_xyt_compute);

/* scat2gridgauss_xy.F */
else if ( !strcmp(name,"scat2gridgauss_xy_init_") ) return (void *)FORTRAN(scat2gridgauss_xy_init);
else if ( !strcmp(name,"scat2gridgauss_xy_work_size_") ) return (void *)FORTRAN(scat2gridgauss_xy_work_size);
else if ( !strcmp(name,"scat2gridgauss_xy_compute_") ) return (void *)FORTRAN(scat2gridgauss_xy_compute);

/* scat2gridgauss_xz.F */
else if ( !strcmp(name,"scat2gridgauss_xz_init_") ) return (void *)FORTRAN(scat2gridgauss_xz_init);
else if ( !strcmp(name,"scat2gridgauss_xz_work_size_") ) return (void *)FORTRAN(scat2gridgauss_xz_work_size);
else if ( !strcmp(name,"scat2gridgauss_xz_compute_") ) return (void *)FORTRAN(scat2gridgauss_xz_compute);

/* scat2gridgauss_yz.F */
else if ( !strcmp(name,"scat2gridgauss_yz_init_") ) return (void *)FORTRAN(scat2gridgauss_yz_init);
else if ( !strcmp(name,"scat2gridgauss_yz_work_size_") ) return (void *)FORTRAN(scat2gridgauss_yz_work_size);
else if ( !strcmp(name,"scat2gridgauss_yz_compute_") ) return (void *)FORTRAN(scat2gridgauss_yz_compute);

/* scat2gridgauss_xt.F */
else if ( !strcmp(name,"scat2gridgauss_xt_init_") ) return (void *)FORTRAN(scat2gridgauss_xt_init);
else if ( !strcmp(name,"scat2gridgauss_xt_work_size_") ) return (void *)FORTRAN(scat2gridgauss_xt_work_size);
else if ( !strcmp(name,"scat2gridgauss_xt_compute_") ) return (void *)FORTRAN(scat2gridgauss_xt_compute);

/* scat2gridgauss_yt.F */
else if ( !strcmp(name,"scat2gridgauss_yt_init_") ) return (void *)FORTRAN(scat2gridgauss_yt_init);
else if ( !strcmp(name,"scat2gridgauss_yt_work_size_") ) return (void *)FORTRAN(scat2gridgauss_yt_work_size);
else if ( !strcmp(name,"scat2gridgauss_yt_compute_") ) return (void *)FORTRAN(scat2gridgauss_yt_compute);

/* scat2gridgauss_zt.F */
else if ( !strcmp(name,"scat2gridgauss_zt_init_") ) return (void *)FORTRAN(scat2gridgauss_zt_init);
else if ( !strcmp(name,"scat2gridgauss_zt_work_size_") ) return (void *)FORTRAN(scat2gridgauss_zt_work_size);
else if ( !strcmp(name,"scat2gridgauss_zt_compute_") ) return (void *)FORTRAN(scat2gridgauss_zt_compute);

/* scat2gridlaplace_xy.F */
else if ( !strcmp(name,"scat2gridlaplace_xy_init_") ) return (void *)FORTRAN(scat2gridlaplace_xy_init);
else if ( !strcmp(name,"scat2gridlaplace_xy_work_size_") ) return (void *)FORTRAN(scat2gridlaplace_xy_work_size);
else if ( !strcmp(name,"scat2gridlaplace_xy_compute_") ) return (void *)FORTRAN(scat2gridlaplace_xy_compute);

/* scat2gridlaplace_xz.F */
else if ( !strcmp(name,"scat2gridlaplace_xz_init_") ) return (void *)FORTRAN(scat2gridlaplace_xz_init);
else if ( !strcmp(name,"scat2gridlaplace_xz_work_size_") ) return (void *)FORTRAN(scat2gridlaplace_xz_work_size);
else if ( !strcmp(name,"scat2gridlaplace_xz_compute_") ) return (void *)FORTRAN(scat2gridlaplace_xz_compute);

/* scat2gridlaplace_yz.F */
else if ( !strcmp(name,"scat2gridlaplace_yz_init_") ) return (void *)FORTRAN(scat2gridlaplace_yz_init);
else if ( !strcmp(name,"scat2gridlaplace_yz_work_size_") ) return (void *)FORTRAN(scat2gridlaplace_yz_work_size);
else if ( !strcmp(name,"scat2gridlaplace_yz_compute_") ) return (void *)FORTRAN(scat2gridlaplace_yz_compute);

/* scat2gridlaplace_xt.F */
else if ( !strcmp(name,"scat2gridlaplace_xt_init_") ) return (void *)FORTRAN(scat2gridlaplace_xt_init);
else if ( !strcmp(name,"scat2gridlaplace_xt_work_size_") ) return (void *)FORTRAN(scat2gridlaplace_xt_work_size);
else if ( !strcmp(name,"scat2gridlaplace_xt_compute_") ) return (void *)FORTRAN(scat2gridlaplace_xt_compute);

/* scat2gridlaplace_yt.F */
else if ( !strcmp(name,"scat2gridlaplace_yt_init_") ) return (void *)FORTRAN(scat2gridlaplace_yt_init);
else if ( !strcmp(name,"scat2gridlaplace_yt_work_size_") ) return (void *)FORTRAN(scat2gridlaplace_yt_work_size);
else if ( !strcmp(name,"scat2gridlaplace_yt_compute_") ) return (void *)FORTRAN(scat2gridlaplace_yt_compute);

/* scat2gridlaplace_zt.F */
else if ( !strcmp(name,"scat2gridlaplace_zt_init_") ) return (void *)FORTRAN(scat2gridlaplace_zt_init);
else if ( !strcmp(name,"scat2gridlaplace_zt_work_size_") ) return (void *)FORTRAN(scat2gridlaplace_zt_work_size);
else if ( !strcmp(name,"scat2gridlaplace_zt_compute_") ) return (void *)FORTRAN(scat2gridlaplace_zt_compute);

/* scat2grid_nobs_xy.F */
else if ( !strcmp(name,"scat2grid_nobs_xy_init_") ) return (void *)FORTRAN(scat2grid_nobs_xy_init);
else if ( !strcmp(name,"scat2grid_nobs_xy_work_size_") ) return (void *)FORTRAN(scat2grid_nobs_xy_work_size);
else if ( !strcmp(name,"scat2grid_nobs_xy_compute_") ) return (void *)FORTRAN(scat2grid_nobs_xy_compute);

else if ( !strcmp(name,"scat2grid_nobs_xyt_init_") ) return (void *)FORTRAN(scat2grid_nobs_xyt_init);
else if ( !strcmp(name,"scat2grid_nobs_xyt_work_size_") ) return (void *)FORTRAN(scat2grid_nobs_xyt_work_size);
else if ( !strcmp(name,"scat2grid_nobs_xyt_compute_") ) return (void *)FORTRAN(scat2grid_nobs_xyt_compute);

/* sorti.F */
else if ( !strcmp(name,"sorti_init_") ) return (void *)FORTRAN(sorti_init);
else if ( !strcmp(name,"sorti_result_limits_") ) return (void *)FORTRAN(sorti_result_limits);
else if ( !strcmp(name,"sorti_work_size_") ) return (void *)FORTRAN(sorti_work_size);
else if ( !strcmp(name,"sorti_compute_") ) return (void *)FORTRAN(sorti_compute);

/* sorti_str.F */
else if ( !strcmp(name,"sorti_str_init_") ) return (void *)FORTRAN(sorti_str_init);
else if ( !strcmp(name,"sorti_str_result_limits_") ) return (void *)FORTRAN(sorti_str_result_limits);
else if ( !strcmp(name,"sorti_str_work_size_") ) return (void *)FORTRAN(sorti_str_work_size);
else if ( !strcmp(name,"sorti_str_compute_") ) return (void *)FORTRAN(sorti_str_compute);

/* sortj.F */
else if ( !strcmp(name,"sortj_init_") ) return (void *)FORTRAN(sortj_init);
else if ( !strcmp(name,"sortj_result_limits_") ) return (void *)FORTRAN(sortj_result_limits);
else if ( !strcmp(name,"sortj_work_size_") ) return (void *)FORTRAN(sortj_work_size);
else if ( !strcmp(name,"sortj_compute_") ) return (void *)FORTRAN(sortj_compute);

/* sortj_str.F */
else if ( !strcmp(name,"sortj_str_init_") ) return (void *)FORTRAN(sortj_str_init);
else if ( !strcmp(name,"sortj_str_result_limits_") ) return (void *)FORTRAN(sortj_str_result_limits);
else if ( !strcmp(name,"sortj_str_work_size_") ) return (void *)FORTRAN(sortj_str_work_size);
else if ( !strcmp(name,"sortj_str_compute_") ) return (void *)FORTRAN(sortj_str_compute);

/* sortk.F */
else if ( !strcmp(name,"sortk_init_") ) return (void *)FORTRAN(sortk_init);
else if ( !strcmp(name,"sortk_result_limits_") ) return (void *)FORTRAN(sortk_result_limits);
else if ( !strcmp(name,"sortk_work_size_") ) return (void *)FORTRAN(sortk_work_size);
else if ( !strcmp(name,"sortk_compute_") ) return (void *)FORTRAN(sortk_compute);

/* sortk_str.F */
else if ( !strcmp(name,"sortk_str_init_") ) return (void *)FORTRAN(sortk_str_init);
else if ( !strcmp(name,"sortk_str_result_limits_") ) return (void *)FORTRAN(sortk_str_result_limits);
else if ( !strcmp(name,"sortk_str_work_size_") ) return (void *)FORTRAN(sortk_str_work_size);
else if ( !strcmp(name,"sortk_str_compute_") ) return (void *)FORTRAN(sortk_str_compute);

/* sortl.F */
else if ( !strcmp(name,"sortl_init_") ) return (void *)FORTRAN(sortl_init);
else if ( !strcmp(name,"sortl_result_limits_") ) return (void *)FORTRAN(sortl_result_limits);
else if ( !strcmp(name,"sortl_work_size_") ) return (void *)FORTRAN(sortl_work_size);
else if ( !strcmp(name,"sortl_compute_") ) return (void *)FORTRAN(sortl_compute);

/* sortl_str.F */
else if ( !strcmp(name,"sortl_str_init_") ) return (void *)FORTRAN(sortl_str_init);
else if ( !strcmp(name,"sortl_str_result_limits_") ) return (void *)FORTRAN(sortl_str_result_limits);
else if ( !strcmp(name,"sortl_str_work_size_") ) return (void *)FORTRAN(sortl_str_work_size);
else if ( !strcmp(name,"sortl_str_compute_") ) return (void *)FORTRAN(sortl_str_compute);

/* sortm.F */
else if ( !strcmp(name,"sortm_init_") ) return (void *)FORTRAN(sortm_init);
else if ( !strcmp(name,"sortm_result_limits_") ) return (void *)FORTRAN(sortm_result_limits);
else if ( !strcmp(name,"sortm_work_size_") ) return (void *)FORTRAN(sortm_work_size);
else if ( !strcmp(name,"sortm_compute_") ) return (void *)FORTRAN(sortm_compute);

/* sortm_str.F */
else if ( !strcmp(name,"sortm_str_init_") ) return (void *)FORTRAN(sortm_str_init);
else if ( !strcmp(name,"sortm_str_result_limits_") ) return (void *)FORTRAN(sortm_str_result_limits);
else if ( !strcmp(name,"sortm_str_work_size_") ) return (void *)FORTRAN(sortm_str_work_size);
else if ( !strcmp(name,"sortm_str_compute_") ) return (void *)FORTRAN(sortm_str_compute);

/* sortn.F */
else if ( !strcmp(name,"sortn_init_") ) return (void *)FORTRAN(sortn_init);
else if ( !strcmp(name,"sortn_result_limits_") ) return (void *)FORTRAN(sortn_result_limits);
else if ( !strcmp(name,"sortn_work_size_") ) return (void *)FORTRAN(sortn_work_size);
else if ( !strcmp(name,"sortn_compute_") ) return (void *)FORTRAN(sortn_compute);

/* sortn_str.F */
else if ( !strcmp(name,"sortn_str_init_") ) return (void *)FORTRAN(sortn_str_init);
else if ( !strcmp(name,"sortn_str_result_limits_") ) return (void *)FORTRAN(sortn_str_result_limits);
else if ( !strcmp(name,"sortn_str_work_size_") ) return (void *)FORTRAN(sortn_str_work_size);
else if ( !strcmp(name,"sortn_str_compute_") ) return (void *)FORTRAN(sortn_str_compute);

/* tauto_cor.F */
else if ( !strcmp(name,"tauto_cor_init_") ) return (void *)FORTRAN(tauto_cor_init);
else if ( !strcmp(name,"tauto_cor_result_limits_") ) return (void *)FORTRAN(tauto_cor_result_limits);
else if ( !strcmp(name,"tauto_cor_work_size_") ) return (void *)FORTRAN(tauto_cor_work_size);
else if ( !strcmp(name,"tauto_cor_compute_") ) return (void *)FORTRAN(tauto_cor_compute);

/* xauto_cor.F */
else if ( !strcmp(name,"xauto_cor_init_") ) return (void *)FORTRAN(xauto_cor_init);
else if ( !strcmp(name,"xauto_cor_result_limits_") ) return (void *)FORTRAN(xauto_cor_result_limits);
else if ( !strcmp(name,"xauto_cor_work_size_") ) return (void *)FORTRAN(xauto_cor_work_size);
else if ( !strcmp(name,"xauto_cor_compute_") ) return (void *)FORTRAN(xauto_cor_compute);

/* eof_space.F */
else if ( !strcmp(name,"eof_space_init_") ) return (void *)FORTRAN(eof_space_init);
else if ( !strcmp(name,"eof_space_result_limits_") ) return (void *)FORTRAN(eof_space_result_limits);
else if ( !strcmp(name,"eof_space_work_size_") ) return (void *)FORTRAN(eof_space_work_size);
else if ( !strcmp(name,"eof_space_compute_") ) return (void *)FORTRAN(eof_space_compute);

/* eof_stat.F */
else if ( !strcmp(name,"eof_stat_init_") ) return (void *)FORTRAN(eof_stat_init);
else if ( !strcmp(name,"eof_stat_result_limits_") ) return (void *)FORTRAN(eof_stat_result_limits);
else if ( !strcmp(name,"eof_stat_work_size_") ) return (void *)FORTRAN(eof_stat_work_size);
else if ( !strcmp(name,"eof_stat_compute_") ) return (void *)FORTRAN(eof_stat_compute);

/* eof_tfunc.F */
else if ( !strcmp(name,"eof_tfunc_init_") ) return (void *)FORTRAN(eof_tfunc_init);
else if ( !strcmp(name,"eof_tfunc_result_limits_") ) return (void *)FORTRAN(eof_tfunc_result_limits);
else if ( !strcmp(name,"eof_tfunc_work_size_") ) return (void *)FORTRAN(eof_tfunc_work_size);
else if ( !strcmp(name,"eof_tfunc_compute_") ) return (void *)FORTRAN(eof_tfunc_compute);

/* eofsvd_space.F */
else if ( !strcmp(name,"eofsvd_space_init_") ) return (void *)FORTRAN(eofsvd_space_init);
else if ( !strcmp(name,"eofsvd_space_result_limits_") ) return (void *)FORTRAN(eofsvd_space_result_limits);
else if ( !strcmp(name,"eofsvd_space_work_size_") ) return (void *)FORTRAN(eofsvd_space_work_size);
else if ( !strcmp(name,"eofsvd_space_compute_") ) return (void *)FORTRAN(eofsvd_space_compute);

/* eofsvd_stat.F */
else if ( !strcmp(name,"eofsvd_stat_init_") ) return (void *)FORTRAN(eofsvd_stat_init);
else if ( !strcmp(name,"eofsvd_stat_result_limits_") ) return (void *)FORTRAN(eofsvd_stat_result_limits);
else if ( !strcmp(name,"eofsvd_stat_work_size_") ) return (void *)FORTRAN(eofsvd_stat_work_size);
else if ( !strcmp(name,"eofsvd_stat_compute_") ) return (void *)FORTRAN(eofsvd_stat_compute);

/* eofsvd_tfunc.F */
else if ( !strcmp(name,"eofsvd_tfunc_init_") ) return (void *)FORTRAN(eofsvd_tfunc_init);
else if ( !strcmp(name,"eofsvd_tfunc_result_limits_") ) return (void *)FORTRAN(eofsvd_tfunc_result_limits);
else if ( !strcmp(name,"eofsvd_tfunc_work_size_") ) return (void *)FORTRAN(eofsvd_tfunc_work_size);
else if ( !strcmp(name,"eofsvd_tfunc_compute_") ) return (void *)FORTRAN(eofsvd_tfunc_compute);

/* compressi.F */
else if ( !strcmp(name,"compressi_init_") ) return (void *)FORTRAN(compressi_init);
else if ( !strcmp(name,"compressi_result_limits_") ) return (void *)FORTRAN(compressi_result_limits);
else if ( !strcmp(name,"compressi_compute_") ) return (void *)FORTRAN(compressi_compute);

/* compressj.F */
else if ( !strcmp(name,"compressj_init_") ) return (void *)FORTRAN(compressj_init);
else if ( !strcmp(name,"compressj_result_limits_") ) return (void *)FORTRAN(compressj_result_limits);
else if ( !strcmp(name,"compressj_compute_") ) return (void *)FORTRAN(compressj_compute);

/* compressk.F */
else if ( !strcmp(name,"compressk_init_") ) return (void *)FORTRAN(compressk_init);
else if ( !strcmp(name,"compressk_result_limits_") ) return (void *)FORTRAN(compressk_result_limits);
else if ( !strcmp(name,"compressk_compute_") ) return (void *)FORTRAN(compressk_compute);

/* compressl.F */
else if ( !strcmp(name,"compressl_init_") ) return (void *)FORTRAN(compressl_init);
else if ( !strcmp(name,"compressl_result_limits_") ) return (void *)FORTRAN(compressl_result_limits);
else if ( !strcmp(name,"compressl_compute_") ) return (void *)FORTRAN(compressl_compute);

/* compressm.F */
else if ( !strcmp(name,"compressm_init_") ) return (void *)FORTRAN(compressm_init);
else if ( !strcmp(name,"compressm_result_limits_") ) return (void *)FORTRAN(compressm_result_limits);
else if ( !strcmp(name,"compressm_compute_") ) return (void *)FORTRAN(compressm_compute);

/* compressn.F */
else if ( !strcmp(name,"compressn_init_") ) return (void *)FORTRAN(compressn_init);
else if ( !strcmp(name,"compressn_result_limits_") ) return (void *)FORTRAN(compressn_result_limits);
else if ( !strcmp(name,"compressn_compute_") ) return (void *)FORTRAN(compressn_compute);

/* compressi_by.F */
else if ( !strcmp(name,"compressi_by_init_") ) return (void *)FORTRAN(compressi_by_init);
else if ( !strcmp(name,"compressi_by_result_limits_") ) return (void *)FORTRAN(compressi_by_result_limits);
else if ( !strcmp(name,"compressi_by_compute_") ) return (void *)FORTRAN(compressi_by_compute);

/* compressj_by.F */
else if ( !strcmp(name,"compressj_by_init_") ) return (void *)FORTRAN(compressj_by_init);
else if ( !strcmp(name,"compressj_by_result_limits_") ) return (void *)FORTRAN(compressj_by_result_limits);
else if ( !strcmp(name,"compressj_by_compute_") ) return (void *)FORTRAN(compressj_by_compute);

/* compressk_by.F */
else if ( !strcmp(name,"compressk_by_init_") ) return (void *)FORTRAN(compressk_by_init);
else if ( !strcmp(name,"compressk_by_result_limits_") ) return (void *)FORTRAN(compressk_by_result_limits);
else if ( !strcmp(name,"compressk_by_compute_") ) return (void *)FORTRAN(compressk_by_compute);

/* compressl_by.F */
else if ( !strcmp(name,"compressl_by_init_") ) return (void *)FORTRAN(compressl_by_init);
else if ( !strcmp(name,"compressl_by_result_limits_") ) return (void *)FORTRAN(compressl_by_result_limits);
else if ( !strcmp(name,"compressl_by_compute_") ) return (void *)FORTRAN(compressl_by_compute);

/* compressm_by.F */
else if ( !strcmp(name,"compressm_by_init_") ) return (void *)FORTRAN(compressm_by_init);
else if ( !strcmp(name,"compressm_by_result_limits_") ) return (void *)FORTRAN(compressm_by_result_limits);
else if ( !strcmp(name,"compressm_by_compute_") ) return (void *)FORTRAN(compressm_by_compute);

/* compressn_by.F */
else if ( !strcmp(name,"compressn_by_init_") ) return (void *)FORTRAN(compressn_by_init);
else if ( !strcmp(name,"compressn_by_result_limits_") ) return (void *)FORTRAN(compressn_by_result_limits);
else if ( !strcmp(name,"compressn_by_compute_") ) return (void *)FORTRAN(compressn_by_compute);

/* box_edges.F */
else if ( !strcmp(name,"box_edges_init_") ) return (void *)FORTRAN(box_edges_init);
else if ( !strcmp(name,"box_edges_result_limits_") ) return (void *)FORTRAN(box_edges_result_limits);
else if ( !strcmp(name,"box_edges_compute_") ) return (void *)FORTRAN(box_edges_compute);


/* labwid.F */
else if ( !strcmp(name,"labwid_init_") ) return (void *)FORTRAN(labwid_init);
else if ( !strcmp(name,"labwid_result_limits_") ) return (void *)FORTRAN(labwid_result_limits);
else if ( !strcmp(name,"labwid_compute_") ) return (void *)FORTRAN(labwid_compute);

/* convolvei.F */
else if ( !strcmp(name,"convolvei_init_") ) return (void *)FORTRAN(convolvei_init);
else if ( !strcmp(name,"convolvei_compute_") ) return (void *)FORTRAN(convolvei_compute);

/* convolvej.F */
else if ( !strcmp(name,"convolvej_init_") ) return (void *)FORTRAN(convolvej_init);
else if ( !strcmp(name,"convolvej_compute_") ) return (void *)FORTRAN(convolvej_compute);

/* convolvek.F */
else if ( !strcmp(name,"convolvek_init_") ) return (void *)FORTRAN(convolvek_init);
else if ( !strcmp(name,"convolvek_compute_") ) return (void *)FORTRAN(convolvek_compute);

/* convolvel.F */
else if ( !strcmp(name,"convolvel_init_") ) return (void *)FORTRAN(convolvel_init);
else if ( !strcmp(name,"convolvel_compute_") ) return (void *)FORTRAN(convolvel_compute);

/* convolvem.F */
else if ( !strcmp(name,"convolvem_init_") ) return (void *)FORTRAN(convolvem_init);
else if ( !strcmp(name,"convolvem_compute_") ) return (void *)FORTRAN(convolvem_compute);

/* convolven.F */
else if ( !strcmp(name,"convolven_init_") ) return (void *)FORTRAN(convolven_init);
else if ( !strcmp(name,"convolven_compute_") ) return (void *)FORTRAN(convolven_compute);

/* curv_range.F */
else if ( !strcmp(name,"curv_range_init_") ) return (void *)FORTRAN(curv_range_init);
else if ( !strcmp(name,"curv_range_result_limits_") ) return (void *)FORTRAN(curv_range_result_limits);
else if ( !strcmp(name,"curv_range_compute_") ) return (void *)FORTRAN(curv_range_compute);

/* curv_to_rect_map.F */
else if ( !strcmp(name,"curv_to_rect_map_init_") ) return (void *)FORTRAN(curv_to_rect_map_init);
else if ( !strcmp(name,"curv_to_rect_map_result_limits_") ) return (void *)FORTRAN(curv_to_rect_map_result_limits);
else if ( !strcmp(name,"curv_to_rect_map_work_size_") ) return (void *)FORTRAN(curv_to_rect_map_work_size);
else if ( !strcmp(name,"curv_to_rect_map_compute_") ) return (void *)FORTRAN(curv_to_rect_map_compute);

/* curv_to_rect.F */
else if ( !strcmp(name,"curv_to_rect_init_") ) return (void *)FORTRAN(curv_to_rect_init);
else if ( !strcmp(name,"curv_to_rect_compute_") ) return (void *)FORTRAN(curv_to_rect_compute);

/* curv_to_rect_fsu.F */
else if ( !strcmp(name,"curv_to_rect_fsu_init_") ) return (void *)FORTRAN(curv_to_rect_fsu_init);
else if ( !strcmp(name,"curv_to_rect_fsu_compute_") ) return (void *)FORTRAN(curv_to_rect_fsu_compute);

/* rect_to_curv.F */
else if ( !strcmp(name,"rect_to_curv_init_") ) return (void *)FORTRAN(rect_to_curv_init);
else if ( !strcmp(name,"rect_to_curv_work_size_") ) return (void *)FORTRAN(rect_to_curv_work_size);
else if ( !strcmp(name,"rect_to_curv_compute_") ) return (void *)FORTRAN(rect_to_curv_compute);

/* date1900.F */
else if ( !strcmp(name,"date1900_init_") ) return (void *)FORTRAN(date1900_init);
else if ( !strcmp(name,"date1900_compute_") ) return (void *)FORTRAN(date1900_compute);

/* days1900toymdhms.F */
else if ( !strcmp(name,"days1900toymdhms_init_") ) return (void *)FORTRAN(days1900toymdhms_init);
else if ( !strcmp(name,"days1900toymdhms_result_limits_") ) return (void *)FORTRAN(days1900toymdhms_result_limits);
else if ( !strcmp(name,"days1900toymdhms_compute_") ) return (void *)FORTRAN(days1900toymdhms_compute);

/* minutes24.F */
else if ( !strcmp(name,"minutes24_init_") ) return (void *)FORTRAN(minutes24_init);
else if ( !strcmp(name,"minutes24_result_limits_") ) return (void *)FORTRAN(minutes24_result_limits);
else if ( !strcmp(name,"minutes24_compute_") ) return (void *)FORTRAN(minutes24_compute);

/* element_index.F */
else if ( !strcmp(name,"element_index_init_") ) return (void *)FORTRAN(element_index_init);
else if ( !strcmp(name,"element_index_compute_") ) return (void *)FORTRAN(element_index_compute);

/* element_index_str.F */
else if ( !strcmp(name,"element_index_str_init_") ) return (void *)FORTRAN(element_index_str_init);
else if ( !strcmp(name,"element_index_str_compute_") ) return (void *)FORTRAN(element_index_str_compute);

/* element_index_str_n.F */
else if ( !strcmp(name,"element_index_str_n_init_") ) return (void *)FORTRAN(element_index_str_n_init);
else if ( !strcmp(name,"element_index_str_n_compute_") ) return (void *)FORTRAN(element_index_str_n_compute);

/* expnd_by_len.F */
else if ( !strcmp(name,"expnd_by_len_init_") ) return (void *)FORTRAN(expnd_by_len_init);
else if ( !strcmp(name,"expnd_by_len_result_limits_") ) return (void *)FORTRAN(expnd_by_len_result_limits);
else if ( !strcmp(name,"expnd_by_len_compute_") ) return (void *)FORTRAN(expnd_by_len_compute);

/* expnd_by_len_str.F */
else if ( !strcmp(name,"expnd_by_len_str_init_") ) return (void *)FORTRAN(expnd_by_len_str_init);
else if ( !strcmp(name,"expnd_by_len_str_result_limits_") ) return (void *)FORTRAN(expnd_by_len_str_result_limits);
else if ( !strcmp(name,"expnd_by_len_str_compute_") ) return (void *)FORTRAN(expnd_by_len_str_compute);

/* expndi_by.F */
else if ( !strcmp(name,"expndi_by_init_") ) return (void *)FORTRAN(expndi_by_init);
else if ( !strcmp(name,"expndi_by_result_limits_") ) return (void *)FORTRAN(expndi_by_result_limits);
else if ( !strcmp(name,"expndi_by_compute_") ) return (void *)FORTRAN(expndi_by_compute);

/* expndi_by_t.F */
else if ( !strcmp(name,"expndi_by_t_init_") ) return (void *)FORTRAN(expndi_by_t_init);
else if ( !strcmp(name,"expndi_by_t_result_limits_") ) return (void *)FORTRAN(expndi_by_t_result_limits);
else if ( !strcmp(name,"expndi_by_t_compute_") ) return (void *)FORTRAN(expndi_by_t_compute);

/* expndi_by_z.F */
else if ( !strcmp(name,"expndi_by_z_init_") ) return (void *)FORTRAN(expndi_by_z_init);
else if ( !strcmp(name,"expndi_by_z_result_limits_") ) return (void *)FORTRAN(expndi_by_z_result_limits);
else if ( !strcmp(name,"expndi_by_z_compute_") ) return (void *)FORTRAN(expndi_by_z_compute);

/* expndi_by_z_counts.F */
else if ( !strcmp(name,"expndi_by_z_counts_init_") ) return (void *)FORTRAN(expndi_by_z_counts_init);
else if ( !strcmp(name,"expndi_by_z_counts_result_limits_") ) return (void *)FORTRAN(expndi_by_z_counts_result_limits);
else if ( !strcmp(name,"expndi_by_z_counts_compute_") ) return (void *)FORTRAN(expndi_by_z_counts_compute);

/* expndi_id_by_z_counts.F */
else if ( !strcmp(name,"expndi_id_by_z_counts_init_") ) return (void *)FORTRAN(expndi_id_by_z_counts_init);
else if ( !strcmp(name,"expndi_id_by_z_counts_result_limits_") ) return (void *)FORTRAN(expndi_id_by_z_counts_result_limits);
else if ( !strcmp(name,"expndi_id_by_z_counts_compute_") ) return (void *)FORTRAN(expndi_id_by_z_counts_compute);

/* expndi_by_m_counts.F */
else if ( !strcmp(name,"expndi_by_m_counts_init_") ) return (void *)FORTRAN(expndi_by_m_counts_init);
else if ( !strcmp(name,"expndi_by_m_counts_compute_") ) return (void *)FORTRAN(expndi_by_m_counts_compute);

/* expndi_by_m_counts_str.F */
else if ( !strcmp(name,"expndi_by_m_counts_str_init_") ) return (void *)FORTRAN(expndi_by_m_counts_str_init);
else if ( !strcmp(name,"expndi_by_m_counts_str_compute_") ) return (void *)FORTRAN(expndi_by_m_counts_str_compute);

/* fc_isubset.F */
else if ( !strcmp(name,"fc_isubset_init_") ) return (void *)FORTRAN(fc_isubset_init);
else if ( !strcmp(name,"fc_isubset_result_limits_") ) return (void *)FORTRAN(fc_isubset_result_limits);
else if ( !strcmp(name,"fc_isubset_custom_axes_") ) return (void *)FORTRAN(fc_isubset_custom_axes);
else if ( !strcmp(name,"fc_isubset_compute_") ) return (void *)FORTRAN(fc_isubset_compute);

/* findhi.F */
else if ( !strcmp(name,"findhi_init_") ) return (void *)FORTRAN(findhi_init);
else if ( !strcmp(name,"findhi_result_limits_") ) return (void *)FORTRAN(findhi_result_limits);
else if ( !strcmp(name,"findhi_work_size_") ) return (void *)FORTRAN(findhi_work_size);
else if ( !strcmp(name,"findhi_compute_") ) return (void *)FORTRAN(findhi_compute);

/* findlo.F */
else if ( !strcmp(name,"findlo_init_") ) return (void *)FORTRAN(findlo_init);
else if ( !strcmp(name,"findlo_result_limits_") ) return (void *)FORTRAN(findlo_result_limits);
else if ( !strcmp(name,"findlo_work_size_") ) return (void *)FORTRAN(findlo_work_size);
else if ( !strcmp(name,"findlo_compute_") ) return (void *)FORTRAN(findlo_compute);

/* is_element_of.F */
else if ( !strcmp(name,"is_element_of_init_") ) return (void *)FORTRAN(is_element_of_init);
else if ( !strcmp(name,"is_element_of_result_limits_") ) return (void *)FORTRAN(is_element_of_result_limits);
else if ( !strcmp(name,"is_element_of_compute_") ) return (void *)FORTRAN(is_element_of_compute);

/* is_element_of_str.F */
else if ( !strcmp(name,"is_element_of_str_init_") ) return (void *)FORTRAN(is_element_of_str_init);
else if ( !strcmp(name,"is_element_of_str_result_limits_") ) return (void *)FORTRAN(is_element_of_str_result_limits);
else if ( !strcmp(name,"is_element_of_str_compute_") ) return (void *)FORTRAN(is_element_of_str_compute);

/* is_element_of_str_n.F */
else if ( !strcmp(name,"is_element_of_str_n_init_") ) return (void *)FORTRAN(is_element_of_str_n_init);
else if ( !strcmp(name,"is_element_of_str_n_result_limits_") ) return (void *)FORTRAN(is_element_of_str_n_result_limits);
else if ( !strcmp(name,"is_element_of_str_n_compute_") ) return (void *)FORTRAN(is_element_of_str_n_compute);

/* lanczos.F */
else if ( !strcmp(name,"lanczos_init_") ) return (void *)FORTRAN(lanczos_init);
else if ( !strcmp(name,"lanczos_work_size_") ) return (void *)FORTRAN(lanczos_work_size);
else if ( !strcmp(name,"lanczos_compute_") ) return (void *)FORTRAN(lanczos_compute);

/* lsl_lowpass.F */
else if ( !strcmp(name,"lsl_lowpass_init_") ) return (void *)FORTRAN(lsl_lowpass_init);
else if ( !strcmp(name,"lsl_lowpass_work_size_") ) return (void *)FORTRAN(lsl_lowpass_work_size);
else if ( !strcmp(name,"lsl_lowpass_compute_") ) return (void *)FORTRAN(lsl_lowpass_compute);

/* scat2grid_t.F */
else if ( !strcmp(name,"scat2grid_t_init_") ) return (void *)FORTRAN(scat2grid_t_init);
else if ( !strcmp(name,"scat2grid_t_work_size_") ) return (void *)FORTRAN(scat2grid_t_work_size);
else if ( !strcmp(name,"scat2grid_t_compute_") ) return (void *)FORTRAN(scat2grid_t_compute);

/* ave_scat2grid_t.F */
else if ( !strcmp(name,"ave_scat2grid_t_init_") ) return (void *)FORTRAN(ave_scat2grid_t_init);
else if ( !strcmp(name,"ave_scat2grid_t_work_size_") ) return (void *)FORTRAN(ave_scat2grid_t_work_size);
else if ( !strcmp(name,"ave_scat2grid_t_compute_") ) return (void *)FORTRAN(ave_scat2grid_t_compute);

/* scat2ddups.F */
else if ( !strcmp(name,"scat2ddups_init_") ) return (void *)FORTRAN(scat2ddups_init);
else if ( !strcmp(name,"scat2ddups_result_limits_") ) return (void *)FORTRAN(scat2ddups_result_limits);
else if ( !strcmp(name,"scat2ddups_compute_") ) return (void *)FORTRAN(scat2ddups_compute);

/* transpose_ef.F */
else if ( !strcmp(name,"transpose_ef_init_") ) return (void *)FORTRAN(transpose_ef_init);
else if ( !strcmp(name,"transpose_ef_result_limits_") ) return (void *)FORTRAN(transpose_ef_result_limits);
else if ( !strcmp(name,"transpose_ef_compute_") ) return (void *)FORTRAN(transpose_ef_compute);

/* transpose_te.F */
else if ( !strcmp(name,"transpose_te_init_") ) return (void *)FORTRAN(transpose_te_init);
else if ( !strcmp(name,"transpose_te_result_limits_") ) return (void *)FORTRAN(transpose_te_result_limits);
else if ( !strcmp(name,"transpose_te_compute_") ) return (void *)FORTRAN(transpose_te_compute);

/* transpose_tf.F */
else if ( !strcmp(name,"transpose_tf_init_") ) return (void *)FORTRAN(transpose_tf_init);
else if ( !strcmp(name,"transpose_tf_result_limits_") ) return (void *)FORTRAN(transpose_tf_result_limits);
else if ( !strcmp(name,"transpose_tf_compute_") ) return (void *)FORTRAN(transpose_tf_compute);

/* transpose_xe.F */
else if ( !strcmp(name,"transpose_xe_init_") ) return (void *)FORTRAN(transpose_xe_init);
else if ( !strcmp(name,"transpose_xe_result_limits_") ) return (void *)FORTRAN(transpose_xe_result_limits);
else if ( !strcmp(name,"transpose_xe_compute_") ) return (void *)FORTRAN(transpose_xe_compute);

/* transpose_xf.F */
else if ( !strcmp(name,"transpose_xf_init_") ) return (void *)FORTRAN(transpose_xf_init);
else if ( !strcmp(name,"transpose_xf_result_limits_") ) return (void *)FORTRAN(transpose_xf_result_limits);
else if ( !strcmp(name,"transpose_xf_compute_") ) return (void *)FORTRAN(transpose_xf_compute);

/* transpose_xt.F */
else if ( !strcmp(name,"transpose_xt_init_") ) return (void *)FORTRAN(transpose_xt_init);
else if ( !strcmp(name,"transpose_xt_result_limits_") ) return (void *)FORTRAN(transpose_xt_result_limits);
else if ( !strcmp(name,"transpose_xt_compute_") ) return (void *)FORTRAN(transpose_xt_compute);

/* transpose_xy.F */
else if ( !strcmp(name,"transpose_xy_init_") ) return (void *)FORTRAN(transpose_xy_init);
else if ( !strcmp(name,"transpose_xy_result_limits_") ) return (void *)FORTRAN(transpose_xy_result_limits);
else if ( !strcmp(name,"transpose_xy_compute_") ) return (void *)FORTRAN(transpose_xy_compute);

/* transpose_xz.F */
else if ( !strcmp(name,"transpose_xz_init_") ) return (void *)FORTRAN(transpose_xz_init);
else if ( !strcmp(name,"transpose_xz_result_limits_") ) return (void *)FORTRAN(transpose_xz_result_limits);
else if ( !strcmp(name,"transpose_xz_compute_") ) return (void *)FORTRAN(transpose_xz_compute);

/* transpose_ye.F */
else if ( !strcmp(name,"transpose_ye_init_") ) return (void *)FORTRAN(transpose_ye_init);
else if ( !strcmp(name,"transpose_ye_result_limits_") ) return (void *)FORTRAN(transpose_ye_result_limits);
else if ( !strcmp(name,"transpose_ye_compute_") ) return (void *)FORTRAN(transpose_ye_compute);

/* transpose_yf.F */
else if ( !strcmp(name,"transpose_yf_init_") ) return (void *)FORTRAN(transpose_yf_init);
else if ( !strcmp(name,"transpose_yf_result_limits_") ) return (void *)FORTRAN(transpose_yf_result_limits);
else if ( !strcmp(name,"transpose_yf_compute_") ) return (void *)FORTRAN(transpose_yf_compute);

/* transpose_yt.F */
else if ( !strcmp(name,"transpose_yt_init_") ) return (void *)FORTRAN(transpose_yt_init);
else if ( !strcmp(name,"transpose_yt_result_limits_") ) return (void *)FORTRAN(transpose_yt_result_limits);
else if ( !strcmp(name,"transpose_yt_compute_") ) return (void *)FORTRAN(transpose_yt_compute);

/* transpose_yz.F */
else if ( !strcmp(name,"transpose_yz_init_") ) return (void *)FORTRAN(transpose_yz_init);
else if ( !strcmp(name,"transpose_yz_result_limits_") ) return (void *)FORTRAN(transpose_yz_result_limits);
else if ( !strcmp(name,"transpose_yz_compute_") ) return (void *)FORTRAN(transpose_yz_compute);

/* transpose_ze.F */
else if ( !strcmp(name,"transpose_ze_init_") ) return (void *)FORTRAN(transpose_ze_init);
else if ( !strcmp(name,"transpose_ze_result_limits_") ) return (void *)FORTRAN(transpose_ze_result_limits);
else if ( !strcmp(name,"transpose_ze_compute_") ) return (void *)FORTRAN(transpose_ze_compute);

/* transpose_zf.F */
else if ( !strcmp(name,"transpose_zf_init_") ) return (void *)FORTRAN(transpose_zf_init);
else if ( !strcmp(name,"transpose_zf_result_limits_") ) return (void *)FORTRAN(transpose_zf_result_limits);
else if ( !strcmp(name,"transpose_zf_compute_") ) return (void *)FORTRAN(transpose_zf_compute);

/* transpose_zt.F */
else if ( !strcmp(name,"transpose_zt_init_") ) return (void *)FORTRAN(transpose_zt_init);
else if ( !strcmp(name,"transpose_zt_result_limits_") ) return (void *)FORTRAN(transpose_zt_result_limits);
else if ( !strcmp(name,"transpose_zt_compute_") ) return (void *)FORTRAN(transpose_zt_compute);

/* xcat.F */
else if ( !strcmp(name,"xcat_init_") ) return (void *)FORTRAN(xcat_init);
else if ( !strcmp(name,"xcat_result_limits_") ) return (void *)FORTRAN(xcat_result_limits);
else if ( !strcmp(name,"xcat_compute_") ) return (void *)FORTRAN(xcat_compute);

/* xcat_str.F */
else if ( !strcmp(name,"xcat_str_init_") ) return (void *)FORTRAN(xcat_str_init);
else if ( !strcmp(name,"xcat_str_result_limits_") ) return (void *)FORTRAN(xcat_str_result_limits);
else if ( !strcmp(name,"xcat_str_compute_") ) return (void *)FORTRAN(xcat_str_compute);

/* ycat.F */
else if ( !strcmp(name,"ycat_init_") ) return (void *)FORTRAN(ycat_init);
else if ( !strcmp(name,"ycat_result_limits_") ) return (void *)FORTRAN(ycat_result_limits);
else if ( !strcmp(name,"ycat_compute_") ) return (void *)FORTRAN(ycat_compute);

/* ycat_str.F */
else if ( !strcmp(name,"ycat_str_init_") ) return (void *)FORTRAN(ycat_str_init);
else if ( !strcmp(name,"ycat_str_result_limits_") ) return (void *)FORTRAN(ycat_str_result_limits);
else if ( !strcmp(name,"ycat_str_compute_") ) return (void *)FORTRAN(ycat_str_compute);

/* zcat.F */
else if ( !strcmp(name,"zcat_init_") ) return (void *)FORTRAN(zcat_init);
else if ( !strcmp(name,"zcat_result_limits_") ) return (void *)FORTRAN(zcat_result_limits);
else if ( !strcmp(name,"zcat_compute_") ) return (void *)FORTRAN(zcat_compute);

/* zcat_str.F */
else if ( !strcmp(name,"zcat_str_init_") ) return (void *)FORTRAN(zcat_str_init);
else if ( !strcmp(name,"zcat_str_result_limits_") ) return (void *)FORTRAN(zcat_str_result_limits);
else if ( !strcmp(name,"zcat_str_compute_") ) return (void *)FORTRAN(zcat_str_compute);

/* tcat.F */
else if ( !strcmp(name,"tcat_init_") ) return (void *)FORTRAN(tcat_init);
else if ( !strcmp(name,"tcat_result_limits_") ) return (void *)FORTRAN(tcat_result_limits);
else if ( !strcmp(name,"tcat_compute_") ) return (void *)FORTRAN(tcat_compute);

/* tcat_str.F */
else if ( !strcmp(name,"tcat_str_init_") ) return (void *)FORTRAN(tcat_str_init);
else if ( !strcmp(name,"tcat_str_result_limits_") ) return (void *)FORTRAN(tcat_str_result_limits);
else if ( !strcmp(name,"tcat_str_compute_") ) return (void *)FORTRAN(tcat_str_compute);

/* ecat.F */
else if ( !strcmp(name,"ecat_init_") ) return (void *)FORTRAN(ecat_init);
else if ( !strcmp(name,"ecat_result_limits_") ) return (void *)FORTRAN(ecat_result_limits);
else if ( !strcmp(name,"ecat_compute_") ) return (void *)FORTRAN(ecat_compute);

/* ecat_str.F */
else if ( !strcmp(name,"ecat_str_init_") ) return (void *)FORTRAN(ecat_str_init);
else if ( !strcmp(name,"ecat_str_result_limits_") ) return (void *)FORTRAN(ecat_str_result_limits);
else if ( !strcmp(name,"ecat_str_compute_") ) return (void *)FORTRAN(ecat_str_compute);

/* fcat.F */
else if ( !strcmp(name,"fcat_init_") ) return (void *)FORTRAN(fcat_init);
else if ( !strcmp(name,"fcat_result_limits_") ) return (void *)FORTRAN(fcat_result_limits);
else if ( !strcmp(name,"fcat_compute_") ) return (void *)FORTRAN(fcat_compute);

/* fcat_str.F */
else if ( !strcmp(name,"fcat_str_init_") ) return (void *)FORTRAN(fcat_str_init);
else if ( !strcmp(name,"fcat_str_result_limits_") ) return (void *)FORTRAN(fcat_str_result_limits);
else if ( !strcmp(name,"fcat_str_compute_") ) return (void *)FORTRAN(fcat_str_compute);

/* xreverse.F */
else if ( !strcmp(name,"xreverse_init_") ) return (void *)FORTRAN(xreverse_init);
else if ( !strcmp(name,"xreverse_result_limits_") ) return (void *)FORTRAN(xreverse_result_limits);
else if ( !strcmp(name,"xreverse_compute_") ) return (void *)FORTRAN(xreverse_compute);

/* yreverse.F */
else if ( !strcmp(name,"yreverse_init_") ) return (void *)FORTRAN(yreverse_init);
else if ( !strcmp(name,"yreverse_result_limits_") ) return (void *)FORTRAN(yreverse_result_limits);
else if ( !strcmp(name,"yreverse_compute_") ) return (void *)FORTRAN(yreverse_compute);

/* zreverse.F */
else if ( !strcmp(name,"zreverse_init_") ) return (void *)FORTRAN(zreverse_init);
else if ( !strcmp(name,"zreverse_result_limits_") ) return (void *)FORTRAN(zreverse_result_limits);
else if ( !strcmp(name,"zreverse_compute_") ) return (void *)FORTRAN(zreverse_compute);

/* treverse.F */
else if ( !strcmp(name,"treverse_init_") ) return (void *)FORTRAN(treverse_init);
else if ( !strcmp(name,"treverse_result_limits_") ) return (void *)FORTRAN(treverse_result_limits);
else if ( !strcmp(name,"treverse_compute_") ) return (void *)FORTRAN(treverse_compute);

/* ereverse.F */
else if ( !strcmp(name,"ereverse_init_") ) return (void *)FORTRAN(ereverse_init);
else if ( !strcmp(name,"ereverse_result_limits_") ) return (void *)FORTRAN(ereverse_result_limits);
else if ( !strcmp(name,"ereverse_compute_") ) return (void *)FORTRAN(ereverse_compute);

/* freverse.F */
else if ( !strcmp(name,"freverse_init_") ) return (void *)FORTRAN(freverse_init);
else if ( !strcmp(name,"freverse_result_limits_") ) return (void *)FORTRAN(freverse_result_limits);
else if ( !strcmp(name,"freverse_compute_") ) return (void *)FORTRAN(freverse_compute);

/* zaxreplace_avg.F */
else if ( !strcmp(name,"zaxreplace_avg_init_") ) return (void *)FORTRAN(zaxreplace_avg_init);
else if ( !strcmp(name,"zaxreplace_avg_work_size_") ) return (void *)FORTRAN(zaxreplace_avg_work_size);
else if ( !strcmp(name,"zaxreplace_avg_compute_") ) return (void *)FORTRAN(zaxreplace_avg_compute);

/* zaxreplace_bin.F */
else if ( !strcmp(name,"zaxreplace_bin_init_") ) return (void *)FORTRAN(zaxreplace_bin_init);
else if ( !strcmp(name,"zaxreplace_bin_work_size_") ) return (void *)FORTRAN(zaxreplace_bin_work_size);
else if ( !strcmp(name,"zaxreplace_bin_compute_") ) return (void *)FORTRAN(zaxreplace_bin_compute);

/* zaxreplace_rev.F */
else if ( !strcmp(name,"zaxreplace_rev_init_") ) return (void *)FORTRAN(zaxreplace_rev_init);
else if ( !strcmp(name,"zaxreplace_rev_compute_") ) return (void *)FORTRAN(zaxreplace_rev_compute);

/* zaxreplace_zlev.F */
else if ( !strcmp(name,"zaxreplace_zlev_init_") ) return (void *)FORTRAN(zaxreplace_zlev_init);
else if ( !strcmp(name,"zaxreplace_zlev_work_size_") ) return (void *)FORTRAN(zaxreplace_zlev_work_size);
else if ( !strcmp(name,"zaxreplace_zlev_compute_") ) return (void *)FORTRAN(zaxreplace_zlev_compute);

/* nco.F */
else if ( !strcmp(name,"nco_init_") ) return (void *)FORTRAN(nco_init);
else if ( !strcmp(name,"nco_result_limits_") ) return (void *)FORTRAN(nco_result_limits);
else if ( !strcmp(name,"nco_compute_") ) return (void *)FORTRAN(nco_compute);

/* nco_attr.F */
else if ( !strcmp(name,"nco_attr_init_") ) return (void *)FORTRAN(nco_attr_init);
else if ( !strcmp(name,"nco_attr_result_limits_") ) return (void *)FORTRAN(nco_attr_result_limits);
else if ( !strcmp(name,"nco_attr_compute_") ) return (void *)FORTRAN(nco_attr_compute);

else if ( !strcmp(name,"tax_datestring_init_") ) return (void *)FORTRAN(tax_datestring_init);
else if ( !strcmp(name,"tax_datestring_compute_") ) return (void *)FORTRAN(tax_datestring_compute);

else if ( !strcmp(name,"tax_day_init_") ) return (void *)FORTRAN(tax_day_init);
else if ( !strcmp(name,"tax_day_compute_") ) return (void *)FORTRAN(tax_day_compute);

else if ( !strcmp(name,"tax_dayfrac_init_") ) return (void *)FORTRAN(tax_dayfrac_init);
else if ( !strcmp(name,"tax_dayfrac_compute_") ) return (void *)FORTRAN(tax_dayfrac_compute);

else if ( !strcmp(name,"tax_jday1900_init_") ) return (void *)FORTRAN(tax_jday1900_init);
else if ( !strcmp(name,"tax_jday1900_compute_") ) return (void *)FORTRAN(tax_jday1900_compute);

else if ( !strcmp(name,"tax_jday_init_") ) return (void *)FORTRAN(tax_jday_init);
else if ( !strcmp(name,"tax_jday_compute_") ) return (void *)FORTRAN(tax_jday_compute);

else if ( !strcmp(name,"tax_month_init_") ) return (void *)FORTRAN(tax_month_init);
else if ( !strcmp(name,"tax_month_work_size_") ) return (void *)FORTRAN(tax_month_work_size);
else if ( !strcmp(name,"tax_month_compute_") ) return (void *)FORTRAN(tax_month_compute);

else if ( !strcmp(name,"tax_times_init_") ) return (void *)FORTRAN(tax_times_init);
else if ( !strcmp(name,"tax_times_compute_") ) return (void *)FORTRAN(tax_times_compute);

else if ( !strcmp(name,"tax_tstep_init_") ) return (void *)FORTRAN(tax_tstep_init);
else if ( !strcmp(name,"tax_tstep_work_size_") ) return (void *)FORTRAN(tax_tstep_work_size);
else if ( !strcmp(name,"tax_tstep_compute_") ) return (void *)FORTRAN(tax_tstep_compute);

else if ( !strcmp(name,"tax_units_init_") ) return (void *)FORTRAN(tax_units_init);
else if ( !strcmp(name,"tax_units_compute_") ) return (void *)FORTRAN(tax_units_compute);

else if ( !strcmp(name,"tax_year_init_") ) return (void *)FORTRAN(tax_year_init);
else if ( !strcmp(name,"tax_year_compute_") ) return (void *)FORTRAN(tax_year_compute);

else if ( !strcmp(name,"tax_yearfrac_init_") ) return (void *)FORTRAN(tax_yearfrac_init);
else if ( !strcmp(name,"tax_yearfrac_compute_") ) return (void *)FORTRAN(tax_yearfrac_compute);

else if ( !strcmp(name,"fill_xy_init_") ) return (void *)FORTRAN(fill_xy_init);
else if ( !strcmp(name,"fill_xy_compute_") ) return (void *)FORTRAN(fill_xy_compute);

else if ( !strcmp(name,"test_opendap_init_") ) return (void *)FORTRAN(test_opendap_init);
else if ( !strcmp(name,"test_opendap_result_limits_") ) return (void *)FORTRAN(test_opendap_result_limits);
else if ( !strcmp(name,"test_opendap_compute_") ) return (void *)FORTRAN(test_opendap_compute);

else if ( !strcmp(name,"unique_str2int_init_") ) return (void *)FORTRAN(unique_str2int_init);
else if ( !strcmp(name,"unique_str2int_compute_") ) return (void *)FORTRAN(unique_str2int_compute);

else if ( !strcmp(name,"bin_index_wt_init_") ) return (void *)FORTRAN(bin_index_wt_init);
else if ( !strcmp(name,"bin_index_wt_result_limits_") ) return (void *)FORTRAN(bin_index_wt_result_limits);
else if ( !strcmp(name,"bin_index_wt_compute_") ) return (void *)FORTRAN(bin_index_wt_compute);

else if ( !strcmp(name,"minmax_init_") ) return (void *)FORTRAN(minmax_init);
else if ( !strcmp(name,"minmax_result_limits_") ) return (void *)FORTRAN(minmax_result_limits);
else if ( !strcmp(name,"minmax_compute_") ) return (void *)FORTRAN(minmax_compute);

else if ( !strcmp(name,"floatstr_init_") ) return (void *)FORTRAN(floatstr_init);
else if ( !strcmp(name,"floatstr_compute_") ) return (void *)FORTRAN(floatstr_compute);

else if ( !strcmp(name,"pt_in_poly_init_") ) return (void *)FORTRAN(pt_in_poly_init);
else if ( !strcmp(name,"pt_in_poly_work_size_") ) return (void *)FORTRAN(pt_in_poly_work_size);
else if ( !strcmp(name,"pt_in_poly_compute_") ) return (void *)FORTRAN(pt_in_poly_compute);

else if ( !strcmp(name,"list_value_xml_init_") ) return (void *)FORTRAN(list_value_xml_init);
else if ( !strcmp(name,"list_value_xml_result_limits_") ) return (void *)FORTRAN(list_value_xml_result_limits);
else if ( !strcmp(name,"list_value_xml_compute_") ) return (void *)FORTRAN(list_value_xml_compute);

else if ( !strcmp(name,"lon_lat_time_string_init_") ) return (void *)FORTRAN(lon_lat_time_string_init);
else if ( !strcmp(name,"lon_lat_time_string_compute_") ) return (void *)FORTRAN(lon_lat_time_string_compute);

else if ( !strcmp(name,"write_webrow_init_") ) return (void *)FORTRAN(write_webrow_init);
else if ( !strcmp(name,"write_webrow_result_limits_") ) return (void *)FORTRAN(write_webrow_result_limits);
else if ( !strcmp(name,"write_webrow_compute_") ) return (void *)FORTRAN(write_webrow_compute);

else if ( !strcmp(name,"str_mask_init_") ) return (void *)FORTRAN(str_mask_init);
else if ( !strcmp(name,"str_mask_compute_") ) return (void *)FORTRAN(str_mask_compute);

else if ( !strcmp(name,"separate_init_") ) return (void *)FORTRAN(separate_init);
else if ( !strcmp(name,"separate_result_limits_") ) return (void *)FORTRAN(separate_result_limits);
else if ( !strcmp(name,"separate_compute_") ) return (void *)FORTRAN(separate_compute);

else if ( !strcmp(name,"separate_str_init_") ) return (void *)FORTRAN(separate_str_init);
else if ( !strcmp(name,"separate_str_result_limits_") ) return (void *)FORTRAN(separate_str_result_limits);
else if ( !strcmp(name,"separate_str_compute_") ) return (void *)FORTRAN(separate_str_compute);

else if ( !strcmp(name,"time_reformat_init_") ) return (void *)FORTRAN(time_reformat_init);
else if ( !strcmp(name,"time_reformat_compute_") ) return (void *)FORTRAN(time_reformat_compute);

else if ( !strcmp(name,"ft_to_orthogonal_init_") ) return (void *)FORTRAN(ft_to_orthogonal_init);
else if ( !strcmp(name,"ft_to_orthogonal_work_size_") ) return (void *)FORTRAN(ft_to_orthogonal_work_size);
else if ( !strcmp(name,"ft_to_orthogonal_compute_") ) return (void *)FORTRAN(ft_to_orthogonal_compute);

else if ( !strcmp(name,"piecewise3_init_") ) return (void *)FORTRAN(piecewise3_init);
else if ( !strcmp(name,"piecewise3_result_limits_") ) return (void *)FORTRAN(piecewise3_result_limits);
else if ( !strcmp(name,"piecewise3_work_size_") ) return (void *)FORTRAN(piecewise3_work_size);
else if ( !strcmp(name,"piecewise3_compute_") ) return (void *)FORTRAN(piecewise3_compute);

else if ( !strcmp(name,"sample_fast_i_init_") ) return (void *)FORTRAN(sample_fast_i_init);
else if ( !strcmp(name,"sample_fast_i_compute_") ) return (void *)FORTRAN(sample_fast_i_compute);

else if ( !strcmp(name,"sample_fast_i_str_init_") ) return (void *)FORTRAN(sample_fast_i_str_init);
else if ( !strcmp(name,"sample_fast_i_str_compute_") ) return (void *)FORTRAN(sample_fast_i_str_compute);

else if ( !strcmp(name,"write_webrow_gwt_init_") ) return (void *)FORTRAN(write_webrow_gwt_init);
else if ( !strcmp(name,"write_webrow_gwt_result_limits_") ) return (void *)FORTRAN(write_webrow_gwt_result_limits);
else if ( !strcmp(name,"write_webrow_gwt_compute_") ) return (void *)FORTRAN(write_webrow_gwt_compute);

/* str_noblanks.F */
else if ( !strcmp(name,"str_noblanks_init_") ) return (void *)FORTRAN(str_noblanks_init);
else if ( !strcmp(name,"str_noblanks_compute_") ) return (void *)FORTRAN(str_noblanks_compute);

/* str_replace.F */
else if ( !strcmp(name,"str_replace_init_") ) return (void *)FORTRAN(str_replace_init);
else if ( !strcmp(name,"str_replace_compute_") ) return (void *)FORTRAN(str_replace_compute);

/* expndi_to_et.F */
else if ( !strcmp(name,"expndi_to_et_init_") ) return (void *)FORTRAN(expndi_to_et_init);
else if ( !strcmp(name,"expndi_to_et_work_size_") ) return (void *)FORTRAN(expndi_to_et_work_size);
else if ( !strcmp(name,"expndi_to_et_compute_") ) return (void *)FORTRAN(expndi_to_et_compute);

/* dot product functions */
else if ( !strcmp(name,"dot_x_init_") ) return (void *)FORTRAN(dot_x_init);
else if ( !strcmp(name,"dot_x_compute_") ) return (void *)FORTRAN(dot_x_compute);

else if ( !strcmp(name,"dot_y_init_") ) return (void *)FORTRAN(dot_y_init);
else if ( !strcmp(name,"dot_y_compute_") ) return (void *)FORTRAN(dot_y_compute);

else if ( !strcmp(name,"dot_z_init_") ) return (void *)FORTRAN(dot_z_init);
else if ( !strcmp(name,"dot_z_compute_") ) return (void *)FORTRAN(dot_z_compute);

else if ( !strcmp(name,"dot_t_init_") ) return (void *)FORTRAN(dot_t_init);
else if ( !strcmp(name,"dot_t_compute_") ) return (void *)FORTRAN(dot_t_compute);

else if ( !strcmp(name,"dot_e_init_") ) return (void *)FORTRAN(dot_e_init);
else if ( !strcmp(name,"dot_e_compute_") ) return (void *)FORTRAN(dot_e_compute);

else if ( !strcmp(name,"dot_f_init_") ) return (void *)FORTRAN(dot_f_init);
else if ( !strcmp(name,"dot_f_compute_") ) return (void *)FORTRAN(dot_f_compute);

/* tracks2grid_mask_ave_xyt.F */
else if ( !strcmp(name,"tracks2grid_mask_ave_xyt_init_") ) return (void *)FORTRAN(tracks2grid_mask_ave_xyt_init);
else if ( !strcmp(name,"tracks2grid_mask_ave_xyt_work_size_") ) return (void *)FORTRAN(tracks2grid_mask_ave_xyt_work_size);
else if ( !strcmp(name,"tracks2grid_mask_ave_xyt_compute_") ) return (void *)FORTRAN(tracks2grid_mask_ave_xyt_compute);

/*  tracks2grid_std_xyt.F */
else if ( !strcmp(name,"tracks2grid_std_xyt_init_") ) return (void *)FORTRAN( tracks2grid_std_xyt_init);
else if ( !strcmp(name,"tracks2grid_std_xyt_work_size_") ) return (void *)FORTRAN( tracks2grid_std_xyt_work_size);
else if ( !strcmp(name,"tracks2grid_std_xyt_compute_") ) return (void *)FORTRAN( tracks2grid_std_xyt_compute);

/*  scat2grid_minmax_xyt.F */
else if ( !strcmp(name,"scat2grid_minmax_xyt_init_") ) return (void *)FORTRAN( scat2grid_minmax_xyt_init);
else if ( !strcmp(name,"scat2grid_minmax_xyt_result_limits_") ) return (void *)FORTRAN( scat2grid_minmax_xyt_result_limits);
else if ( !strcmp(name,"scat2grid_minmax_xyt_work_size_") ) return (void *)FORTRAN( scat2grid_minmax_xyt_work_size);
else if ( !strcmp(name,"scat2grid_minmax_xyt_compute_") ) return (void *)FORTRAN( scat2grid_minmax_xyt_compute);

/*  scat2grid_std_xyt.F */
else if ( !strcmp(name,"scat2grid_std_xyt_init_") ) return (void *)FORTRAN( scat2grid_std_xyt_init);
else if ( !strcmp(name,"scat2grid_std_xyt_work_size_") ) return (void *)FORTRAN( scat2grid_std_xyt_work_size);
else if ( !strcmp(name,"scat2grid_std_xyt_compute_") ) return (void *)FORTRAN( scat2grid_std_xyt_compute);

/*  earth_distance.F */
else if ( !strcmp(name,"earth_distance_init_") ) return (void *)FORTRAN( earth_distance_init);
else if ( !strcmp(name,"earth_distance_compute_") ) return (void *)FORTRAN( earth_distance_compute);

return NULL;
 }
/*  End of function pointer list for internally-linked External Functions
 *  ------------------------------------ */
