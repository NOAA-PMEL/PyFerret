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
* V6.12 *acm* 8/07 add functions scat2grid_bin_xy and scatgrid_nobs_xy.F


/* .................... Includes .................... */
 
/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build*/ 
#include <wchar.h>
#include <unistd.h>		/* for convenience */
#include <stdlib.h>		/* for convenience */
#include <stdio.h>		/* for convenience */
#include <string.h>		/* for convenience */
#include <fcntl.h>		/* for fcntl() */
#include <dlfcn.h>		/* for dynamic linking */
#include <signal.h>             /* for signal() */
#include <setjmp.h>             /* required for jmp_buf */

#include <sys/types.h>	        /* required for some of our prototypes */
#include <sys/stat.h>
#include <sys/errno.h>

#include "EF_Util.h"
#include "list.h"  /* locally added list library */


/* ................ Global Variables ................ */
/*
 * The memory_ptr, mr_list_ptr and cx_list_ptr are obtained from Ferret
 * and cached whenever they are passed into one of the "efcn_" functions.
 * These pointers can be accessed by the utility functions in efn_ext/.
 * This way the EF writer does not need to see these pointers.
 */

static LIST  *GLOBAL_ExternalFunctionList;
float *GLOBAL_memory_ptr;
int   *GLOBAL_mr_list_ptr;
int   *GLOBAL_cx_list_ptr;
int   *GLOBAL_mres_ptr;
float *GLOBAL_bad_flag_ptr;

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
static int I_have_warned_already = TRUE; /* Warning turned off Jan '98 */


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

int  FORTRAN(efcn_gather_info)( int * );
void FORTRAN(efcn_get_custom_axes)( int *, int *, int * );
void FORTRAN(efcn_get_result_limits)( int *, float *, int *, int *, int * );
void FORTRAN(efcn_compute)( int *, int *, int *, int *, int *, float *, int *, float *, int * );


void FORTRAN(efcn_get_custom_axis_sub)( int *, int *, double *, double *, double *, char *, int * );

int  FORTRAN(efcn_get_id)( char * );
int  FORTRAN(efcn_match_template)( int *, char * );

void FORTRAN(efcn_get_name)( int *, char * );
void FORTRAN(efcn_get_version)( int *, float * );
void FORTRAN(efcn_get_descr)( int *, char * );
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


/* .... Functions called internally .... */

/* Fortran routines from the efn/ directory */
void FORTRAN(efcn_copy_array_dims)(void);
void FORTRAN(efcn_set_work_array_dims)(int *, int *, int *, int *, int *, int *, int *, int *, int *);
void FORTRAN(efcn_get_workspace_addr)(float *, int *, float *);

static void EF_signal_handler(int);
static void (*fpe_handler)(int);      /* function pointers */
static void (*segv_handler)(int);
static void (*int_handler)(int);
static void (*bus_handler)(int);
int EF_Util_setsig();
int EF_Util_ressig();


void FORTRAN(ef_err_bail_out)(int *, char *);

void EF_store_globals(float *, int *, int *, int *, float *);

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

void FORTRAN(ffta_init)(int *);
void FORTRAN(ffta_custom_axes)(int *);
void FORTRAN(ffta_result_limits)(int *);
void FORTRAN(ffta_work_size)(int *);
void FORTRAN(ffta_compute)(int *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(fftp_init)(int *);
void FORTRAN(fftp_custom_axes)(int *);
void FORTRAN(fftp_result_limits)(int *);
void FORTRAN(fftp_work_size)(int *);
void FORTRAN(fftp_compute)(int *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(fft_im_init)(int *);
void FORTRAN(fft_im_custom_axes)(int *);
void FORTRAN(fft_im_result_limits)(int *);
void FORTRAN(fft_im_work_size)(int *);
void FORTRAN(fft_im_compute)(int *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(fft_inverse_init)(int *);
void FORTRAN(fft_inverse_result_limits)(int *);
void FORTRAN(fft_inverse_work_size)(int *);
void FORTRAN(fft_inverse_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *);

void FORTRAN(fft_re_init)(int *);
void FORTRAN(fft_re_custom_axes)(int *);
void FORTRAN(fft_re_result_limits)(int *);
void FORTRAN(fft_re_work_size)(int *);
void FORTRAN(fft_re_compute)(int *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(sampleij_init)(int *);
void FORTRAN(sampleij_result_limits)(int *);
void FORTRAN(sampleij_work_size)(int *);
void FORTRAN(sampleij_compute)(int *, float *, float *, float *, 
       float *, float *, float *);

void FORTRAN(samplet_date_init)(int *);
void FORTRAN(samplet_date_result_limits)(int *);
void FORTRAN(samplet_date_work_size)(int *);
void FORTRAN(samplet_date_compute)(int *, float *, float *,
      float *, float *, float *, float *, float *, float *, 
      float *, float *);

void FORTRAN(samplexy_init)(int *);
void FORTRAN(samplexy_result_limits)(int *);
void FORTRAN(samplexy_work_size)(int *);
void FORTRAN(samplexy_compute)(int *, float *, float *,
      float *, float *, float *, float *);

void FORTRAN(scat2gridgauss_xy_init)(int *);
void FORTRAN(scat2gridgauss_xy_work_size)(int *);
void FORTRAN(scat2gridgauss_xy_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(scat2gridgauss_xz_init)(int *);
void FORTRAN(scat2gridgauss_xz_work_size)(int *);
void FORTRAN(scat2gridgauss_xz_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(scat2gridgauss_yz_init)(int *);
void FORTRAN(scat2gridgauss_yz_work_size)(int *);
void FORTRAN(scat2gridgauss_yz_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(scat2gridgauss_xt_init)(int *);
void FORTRAN(scat2gridgauss_xt_work_size)(int *);
void FORTRAN(scat2gridgauss_xt_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(scat2gridgauss_yt_init)(int *);
void FORTRAN(scat2gridgauss_yt_work_size)(int *);
void FORTRAN(scat2gridgauss_yt_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(scat2gridgauss_zt_init)(int *);
void FORTRAN(scat2gridgauss_zt_work_size)(int *);
void FORTRAN(scat2gridgauss_zt_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *, float *);

void FORTRAN(scat2gridlaplace_xy_init)(int *);
void FORTRAN(scat2gridlaplace_xy_work_size)(int *);
void FORTRAN(scat2gridlaplace_xy_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *);

void FORTRAN(scat2gridlaplace_xz_init)(int *);
void FORTRAN(scat2gridlaplace_xz_work_size)(int *);
void FORTRAN(scat2gridlaplace_xz_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *);

void FORTRAN(scat2gridlaplace_yz_init)(int *);
void FORTRAN(scat2gridlaplace_yz_work_size)(int *);
void FORTRAN(scat2gridlaplace_yz_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *);


void FORTRAN(scat2gridlaplace_xt_init)(int *);
void FORTRAN(scat2gridlaplace_xt_work_size)(int *);
void FORTRAN(scat2gridlaplace_xt_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *);

void FORTRAN(scat2gridlaplace_yt_init)(int *);
void FORTRAN(scat2gridlaplace_yt_work_size)(int *);
void FORTRAN(scat2gridlaplace_yt_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *);

void FORTRAN(scat2gridlaplace_zt_init)(int *);
void FORTRAN(scat2gridlaplace_zt_work_size)(int *);
void FORTRAN(scat2gridlaplace_zt_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, float *, 
                           float *, float *);


void FORTRAN(sorti_init)(int *);
void FORTRAN(sorti_result_limits)(int *);
void FORTRAN(sorti_work_size)(int *);
void FORTRAN(sorti_compute)(int *, float *, float *, 
      float *, float *);
                   
void FORTRAN(sortj_init)(int *);
void FORTRAN(sortj_result_limits)(int *);
void FORTRAN(sortj_work_size)(int *);
void FORTRAN(sortj_compute)(int *, float *, float *, 
      float *, float *);

void FORTRAN(sortk_init)(int *);
void FORTRAN(sortk_result_limits)(int *);
void FORTRAN(sortk_work_size)(int *);
void FORTRAN(sortk_compute)(int *, float *, float *, 
      float *, float *);

void FORTRAN(sortl_init)(int *);
void FORTRAN(sortl_result_limits)(int *);
void FORTRAN(sortl_work_size)(int *);
void FORTRAN(sortl_compute)(int *, float *, float *, 
      float *, float *);

void FORTRAN(tauto_cor_init)(int *);
void FORTRAN(tauto_cor_result_limits)(int *);
void FORTRAN(tauto_cor_work_size)(int *);
void FORTRAN(tauto_cor_compute)(int *, float *, float *, float *, 
                           float *, float *);

void FORTRAN(xauto_cor_init)(int *);
void FORTRAN(xauto_cor_result_limits)(int *);
void FORTRAN(xauto_cor_work_size)(int *);
void FORTRAN(xauto_cor_compute)(int *, float *, float *, float *, 
                           float *, float *);
						   
void FORTRAN(eof_space_init)(int *);
void FORTRAN(eof_space_result_limits)(int *);
void FORTRAN(eof_space_work_size)(int *);
void FORTRAN(eof_space_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, 
                           float *, float *, float *, float *, float *);
						   
void FORTRAN(eof_stat_init)(int *);
void FORTRAN(eof_stat_result_limits)(int *);
void FORTRAN(eof_stat_work_size)(int *);
void FORTRAN(eof_stat_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, 
                           float *, float *, float *, float *, float *);
						   
void FORTRAN(eof_tfunc_init)(int *);
void FORTRAN(eof_tfunc_result_limits)(int *);
void FORTRAN(eof_tfunc_work_size)(int *);
void FORTRAN(eof_tfunc_compute)(int *, float *, float *, float *, 
                           float *, float *, float *, float *, 
                           float *, float *, float *, float *, float *);
 
void FORTRAN(compressi_init)(int *);
void FORTRAN(compressi_result_limits)(int *);
void FORTRAN(compressi_compute)(int *, float *, float *);

void FORTRAN(compressj_init)(int *);
void FORTRAN(compressj_result_limits)(int *);
void FORTRAN(compressj_compute)(int *, float *, float *);

void FORTRAN(compressk_init)(int *);
void FORTRAN(compressk_result_limits)(int *);
void FORTRAN(compressk_compute)(int *, float *, float *);

void FORTRAN(compressl_init)(int *);
void FORTRAN(compressl_result_limits)(int *);
void FORTRAN(compressl_compute)(int *, float *, float *);

void FORTRAN(compressi_by_init)(int *);
void FORTRAN(compressi_by_result_limits)(int *);
void FORTRAN(compressi_by_compute)(int *, float *, float *);

void FORTRAN(compressj_by_init)(int *);
void FORTRAN(compressj_by_result_limits)(int *);
void FORTRAN(compressj_by_compute)(int *, float *, float *);

void FORTRAN(compressk_by_init)(int *);
void FORTRAN(compressk_by_result_limits)(int *);
void FORTRAN(compressk_by_compute)(int *, float *, float *);

void FORTRAN(compressl_by_init)(int *);
void FORTRAN(compressl_by_result_limits)(int *);
void FORTRAN(compressl_by_compute)(int *, float *, float *);

void FORTRAN(labwid_init)(int *);
void FORTRAN(labwid_result_limits)(int *);
void FORTRAN(labwid_compute)(int *, float *, float *);

void FORTRAN(convolvei_init)(int *);
void FORTRAN(convolvei_compute)(int *, float *, float *, float *);

void FORTRAN(convolvej_init)(int *);
void FORTRAN(convolvej_compute)(int *, float *, float *, float *);

void FORTRAN(convolvek_init)(int *);
void FORTRAN(convolvek_compute)(int *, float *, float *, float *);

void FORTRAN(convolvel_init)(int *);
void FORTRAN(convolvel_compute)(int *, float *, float *, float *);

void FORTRAN(curv_range_init)(int *);
void FORTRAN(curv_range_result_limits)(int *);
void FORTRAN(curv_range_compute)(int *, float *, float *, float *, float *, float *, float *, float *, float *);

void FORTRAN(curv_to_rect_map_init)(int *);
void FORTRAN(curv_to_rect_map_result_limits)(int *);
void FORTRAN(curv_to_rect_map_work_size)(int *);
void FORTRAN(curv_to_rect_map_compute)(int *, float *, float *, float *, float *, float *, 
                                       float *, float *, float *, float *, float *, float *, float *, float *);
void FORTRAN(curv_to_rect_init)(int *);
void FORTRAN(curv_to_rect_compute)(int *, float *, float *, float *);

void FORTRAN(rect_to_curv_init)(int *);
void FORTRAN(rect_to_curv_work_size)(int *);
void FORTRAN(rect_to_curv_compute)(int *, float *, float *, float *, float *, float *, 
                                       float *, float *, float *, float *, float *, float *, float *, float *);

void FORTRAN(date1900_init)(int *);
void FORTRAN(date1900_result_limits)(int *);
void FORTRAN(date1900_compute)(int *, float *, float *);

void FORTRAN(days1900toymdhms_init)(int *);
void FORTRAN(days1900toymdhms_result_limits)(int *);
void FORTRAN(days1900toymdhms_compute)(int *, float *, float *);

void FORTRAN(minutes24_init)(int *);
void FORTRAN(minutes24_result_limits)(int *);
void FORTRAN(minutes24_compute)(int *, float *, float *);

void FORTRAN(element_index_init)(int *);
void FORTRAN(element_index_compute)(int *, float *, float *);

void FORTRAN(element_index_str_init)(int *);
void FORTRAN(element_index_str_compute)(int *, float *, float *);

void FORTRAN(expndi_by_init)(int *);
void FORTRAN(expndi_by_result_limits)(int *);
void FORTRAN(expndi_by_compute)(int *, float *, float *, float *, float *);

void FORTRAN(expndi_by_t_init)(int *);
void FORTRAN(expndi_by_t_result_limits)(int *);
void FORTRAN(expndi_by_t_compute)(int *, float *, float *, float *, float *, float *);

void FORTRAN(expndi_by_z_init)(int *);
void FORTRAN(expndi_by_z_result_limits)(int *);
void FORTRAN(expndi_by_z_compute)(int *, float *, float *, float *, float *, float *);

void FORTRAN(findhi_init)(int *);
void FORTRAN(findhi_result_limits)(int *);
void FORTRAN(findhi_work_size)(int *);
void FORTRAN(findhi_compute)(int *, float *, float *, float *, float *, 
                            float *, float *, float *, float *);

void FORTRAN(findlo_init)(int *);
void FORTRAN(findlo_result_limits)(int *);
void FORTRAN(findlo_work_size)(int *);
void FORTRAN(findlo_compute)(int *, float *, float *, float *, float *, 
                            float *, float *, float *, float *);

void FORTRAN(is_element_of_init)(int *);
void FORTRAN(is_element_of_result_limits)(int *);
void FORTRAN(is_element_of_compute)(int *, float *, float *, float *);

void FORTRAN(is_element_of_str_init)(int *);
void FORTRAN(is_element_of_str_result_limits)(int *);
void FORTRAN(is_element_of_str_compute)(int *, float *, float *, float *);

void FORTRAN(lanczos_init)(int *);
void FORTRAN(lanczos_work_size)(int *);
void FORTRAN(lanczos_compute)(int *, float *, float *, float *, float *, 
                            float *, float *);

void FORTRAN(lsl_lowpass_init)(int *);
void FORTRAN(lsl_lowpass_work_size)(int *);
void FORTRAN(lsl_lowpass_compute)(int *, float *, float *, float *, float *, 
                            float *, float *, float *, float *);
							

void FORTRAN(samplexy_curv_init)(int *);
void FORTRAN(samplexy_curv_result_limits)(int *);
void FORTRAN(samplexy_curv_work_size)(int *);
void FORTRAN(samplexy_curv_compute)(int *, float *, float *,
      float *, float *, float *, float *, float *);

void FORTRAN(samplexy_curv_avg_init)(int *);
void FORTRAN(samplexy_curv_avg_result_limits)(int *);
void FORTRAN(samplexy_curv_avg_work_size)(int *);
void FORTRAN(samplexy_curv_avg_compute)(int *, float *, float *,
      float *, float *, float *, float *, float *);

void FORTRAN(samplexy_curv_nrst_init)(int *);
void FORTRAN(samplexy_curv_nrst_result_limits)(int *);
void FORTRAN(samplexy_curv_nrst_work_size)(int *);
void FORTRAN(samplexy_curv_nrst_compute)(int *, float *, float *,
      float *, float *, float *, float *, float *);

void FORTRAN(samplexy_closest_init)(int *);
void FORTRAN(samplexy_closest_result_limits)(int *);
void FORTRAN(samplexy_closest_work_size)(int *);
void FORTRAN(samplexy_closest_compute)(int *, float *, float *,
      float *, float *, float *, float *);

void FORTRAN(samplexz_init)(int *);
void FORTRAN(samplexz_result_limits)(int *);
void FORTRAN(samplexz_work_size)(int *);
void FORTRAN(samplexz_compute)(int *, float *, float *,
      float *, float *, float *, float *);

void FORTRAN(sampleyz_init)(int *);
void FORTRAN(sampleyz_result_limits)(int *);
void FORTRAN(sampleyz_work_size)(int *);
void FORTRAN(sampleyz_compute)(int *, float *, float *,
      float *, float *, float *, float *);

void FORTRAN(scat2ddups_init)(int *);
void FORTRAN(scat2ddups_result_limits)(int *);
void FORTRAN(scat2ddups_compute)(int *, float *, float *, float *, float *, float *);

void FORTRAN(ave_scat2grid_t_init)(int *);
void FORTRAN(ave_scat2grid_t_work_size)(int *);
void FORTRAN(ave_scat2grid_t_compute)(int *, float *, float *,
      float *, float *, float *, float *);

void FORTRAN(scat2grid_t_init)(int *);
void FORTRAN(scat2grid_t_work_size)(int *);
void FORTRAN(scat2grid_t_compute)(int *, float *, float *, float *, float *);

void FORTRAN(transpose_xt_init)(int *);
void FORTRAN(transpose_xt_result_limits)(int *);
void FORTRAN(transpose_xt_compute)(int *, float *, float *);

void FORTRAN(transpose_xy_init)(int *);
void FORTRAN(transpose_xy_result_limits)(int *);
void FORTRAN(transpose_xy_compute)(int *, float *, float *);

void FORTRAN(transpose_xz_init)(int *);
void FORTRAN(transpose_xz_result_limits)(int *);
void FORTRAN(transpose_xz_compute)(int *, float *, float *);

void FORTRAN(transpose_yt_init)(int *);
void FORTRAN(transpose_yt_result_limits)(int *);
void FORTRAN(transpose_yt_compute)(int *, float *, float *);

void FORTRAN(transpose_yz_init)(int *);
void FORTRAN(transpose_yz_result_limits)(int *);
void FORTRAN(transpose_yz_compute)(int *, float *, float *);

void FORTRAN(transpose_zt_init)(int *);
void FORTRAN(transpose_zt_result_limits)(int *);
void FORTRAN(transpose_zt_compute)(int *, float *, float *);

void FORTRAN(xcat_init)(int *);
void FORTRAN(xcat_result_limits)(int *);
void FORTRAN(xcat_compute)(int *, float *, float *, float *);

void FORTRAN(ycat_init)(int *);
void FORTRAN(ycat_result_limits)(int *);
void FORTRAN(ycat_compute)(int *, float *, float *, float *);

void FORTRAN(zcat_init)(int *);
void FORTRAN(zcat_result_limits)(int *);
void FORTRAN(zcat_compute)(int *, float *, float *, float *);

void FORTRAN(tcat_init)(int *);
void FORTRAN(tcat_result_limits)(int *);
void FORTRAN(tcat_compute)(int *, float *, float *, float *);

void FORTRAN(xreverse_init)(int *);
void FORTRAN(xreverse_result_limits)(int *);
void FORTRAN(xreverse_compute)(int *, float *, float *);

void FORTRAN(yreverse_init)(int *);
void FORTRAN(yreverse_result_limits)(int *);
void FORTRAN(yreverse_compute)(int *, float *, float *);

void FORTRAN(zreverse_init)(int *);
void FORTRAN(zreverse_result_limits)(int *);
void FORTRAN(zreverse_compute)(int *, float *, float *);

void FORTRAN(treverse_init)(int *);
void FORTRAN(treverse_result_limits)(int *);
void FORTRAN(treverse_compute)(int *, float *, float *);

void FORTRAN(zaxreplace_avg_init)(int *);
void FORTRAN(zaxreplace_avg_work_size)(int *);
void FORTRAN(zaxreplace_avg_compute)(int *, float *, float *, float *, 
             float *, float *, float *, float *, float *, float *, float *, float *);

void FORTRAN(zaxreplace_bin_init)(int *);
void FORTRAN(zaxreplace_bin_work_size)(int *);
void FORTRAN(zaxreplace_bin_compute)(int *, float *, float *, float *, 
             float *, float *, float *, float *, float *, float *, float *, float *);

void FORTRAN(zaxreplace_rev_init)(int *);
void FORTRAN(zaxreplace_rev_work_size)(int *);
void FORTRAN(zaxreplace_rev_compute)(int *, float *, float *, float *, 
             float *, float *, float *, float *, float *, float *, float *, float *);

void FORTRAN(zaxreplace_zlev_init)(int *);
void FORTRAN(zaxreplace_zlev_work_size)(int *);
void FORTRAN(zaxreplace_zlev_compute)(int *, float *, float *, float *, float *, float *);

void FORTRAN(nco_attr_init)(int *);
void FORTRAN(nco_attr_result_limits)(int *);
void FORTRAN(nco_attr_compute)(int *, float *, float *, float *);

void FORTRAN(nco_init)(int *);
void FORTRAN(nco_result_limits)(int *);
void FORTRAN(nco_compute)(int *, float *, float *, float *);


void FORTRAN(tax_datestring_init)(int *);
void FORTRAN(tax_datestring_compute)(int *, float *, float *, float *, float *);

void FORTRAN(tax_day_init)(int *);
void FORTRAN(tax_day_compute)(int *, float *, float *, float *);

void FORTRAN(tax_dayfrac_init)(int *);
void FORTRAN(tax_dayfrac_compute)(int *, float *, float *, float *);

void FORTRAN(tax_jday1900_init)(int *);
void FORTRAN(tax_jday1900_compute)(int *, float *, float *, float *);

void FORTRAN(tax_jday_init)(int *);
void FORTRAN(tax_jday_compute)(int *, float *, float *, float *);

void FORTRAN(tax_month_init)(int *);
void FORTRAN(tax_month_compute)(int *, float *, float *, float *);

void FORTRAN(tax_times_init)(int *);
void FORTRAN(tax_times_compute)(int *, float *, float *);

void FORTRAN(tax_tstep_init)(int *);
void FORTRAN(tax_tstep_compute)(int *, float *, float *, float *);

void FORTRAN(tax_units_init)(int *);
void FORTRAN(tax_units_compute)(int *, float *, float*);

void FORTRAN(tax_year_init)(int *);
void FORTRAN(tax_year_compute)(int *, float *, float *, float *);

void FORTRAN(tax_yearfrac_init)(int *);
void FORTRAN(tax_yearfrac_compute)(int *, float *, float *, float *);

void FORTRAN(fill_xy_init)(int *);
void FORTRAN(fill_xy_compute)(int *, float *, float *, float *, float *);

void FORTRAN(test_opendap_init)(int *);
void FORTRAN(test_opendap_result_limits)(int *);
void FORTRAN(test_opendap_compute)(int *, float *, float *);

void FORTRAN(scatgrid_nobs_xy_init)(int *);
void FORTRAN(scatgrid_nobs_xy_work_size)(int *);
void FORTRAN(scatgrid_nobs_xy_compute)(int *, float *, float *);

void FORTRAN(scat2grid_bin_xy_init)(int *);
void FORTRAN(scat2grid_bin_xy_work_size)(int *);
void FORTRAN(scat2grid_bin_xy_compute)(int *, float *, float *);

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

/*
 * Find all of the ~.so files in directories listed in the
 * FER_EXTERNAL_FUNCTIONS environment variable and add all 
 * the names and associated directory information to the 
 * GLOBAL_ExternalFunctionList.
 */
int FORTRAN(efcn_scan)( int *gfcn_num_internal )
{
  
  FILE *file_ptr=NULL;
  ExternalFunction ef; 
 
  char file[EF_MAX_NAME_LENGTH]="";
  char *path_ptr=NULL, path[8192]="";
  char paths[8192]="", cmd[EF_MAX_DESCRIPTION_LENGTH]="";
  int count=0, status=LIST_OK;
  int i_intEF; int N_INTEF;

  static int return_val=0; /* static because it needs to exist after the return statement */
    
/*  ------------------------------------
 *  Count and list the names of internally linked EF's
 *  Lines with names are generated by the perl script 
 *  int_dlsym.pl.  Check that N_INTEF is correctly defined below.
 */

  /*  *******NOTE******
      *kob*  6/01 -  The below initialization code should really be in
      it's own, separate c routine.  So, the next time and internal 
      external function is added, please move the code to it's own routine */

#define N_INTEF 104

struct {
  char funcname[EF_MAX_NAME_LENGTH];
} I_EFnames[N_INTEF];

   strcpy(I_EFnames[0].funcname, "ave_scat2grid_t");
   strcpy(I_EFnames[1].funcname, "compressi");
   strcpy(I_EFnames[2].funcname, "compressj");
   strcpy(I_EFnames[3].funcname, "compressk");
   strcpy(I_EFnames[4].funcname, "compressl");
   strcpy(I_EFnames[5].funcname, "compressi_by");
   strcpy(I_EFnames[6].funcname, "compressj_by");
   strcpy(I_EFnames[7].funcname, "compressk_by");
   strcpy(I_EFnames[8].funcname, "compressl_by");
   strcpy(I_EFnames[9].funcname, "convolvei");
   strcpy(I_EFnames[10].funcname, "convolvej");
   strcpy(I_EFnames[11].funcname, "convolvek");
   strcpy(I_EFnames[12].funcname, "convolvel");
   strcpy(I_EFnames[13].funcname, "curv_range");
   strcpy(I_EFnames[14].funcname, "curv_to_rect_map");
   strcpy(I_EFnames[15].funcname, "curv_to_rect");
   strcpy(I_EFnames[16].funcname, "date1900");
   strcpy(I_EFnames[17].funcname, "days1900toymdhms");
   strcpy(I_EFnames[18].funcname, "element_index");
   strcpy(I_EFnames[19].funcname, "element_index_str");
   strcpy(I_EFnames[20].funcname, "eof_stat");
   strcpy(I_EFnames[21].funcname, "eof_tfunc");
   strcpy(I_EFnames[22].funcname, "eof_space");
   strcpy(I_EFnames[23].funcname, "expndi_by");
   strcpy(I_EFnames[24].funcname, "expndi_by_t");
   strcpy(I_EFnames[25].funcname, "expndi_by_z");
   strcpy(I_EFnames[26].funcname, "ffta");
   strcpy(I_EFnames[27].funcname, "fftp");
   strcpy(I_EFnames[28].funcname, "fft_im");
   strcpy(I_EFnames[29].funcname, "fft_inverse");
   strcpy(I_EFnames[30].funcname, "fft_re");
   strcpy(I_EFnames[31].funcname, "fill_xy");
   strcpy(I_EFnames[32].funcname, "findhi");
   strcpy(I_EFnames[33].funcname, "findlo");
   strcpy(I_EFnames[34].funcname, "is_element_of");
   strcpy(I_EFnames[35].funcname, "is_element_of_str");
   strcpy(I_EFnames[36].funcname, "labwid");
   strcpy(I_EFnames[37].funcname, "lanczos");
   strcpy(I_EFnames[38].funcname, "lsl_lowpass");
   strcpy(I_EFnames[39].funcname, "minutes24");
   strcpy(I_EFnames[40].funcname, "nco");
   strcpy(I_EFnames[41].funcname, "nco_attr");
   strcpy(I_EFnames[42].funcname, "rect_to_curv");
   strcpy(I_EFnames[43].funcname, "sampleij");
   strcpy(I_EFnames[44].funcname, "samplet_date");
   strcpy(I_EFnames[45].funcname, "samplexy");
   strcpy(I_EFnames[46].funcname, "samplexy_closest");
   strcpy(I_EFnames[47].funcname, "samplexy_curv");
   strcpy(I_EFnames[48].funcname, "samplexy_curv_avg");
   strcpy(I_EFnames[49].funcname, "samplexy_curv_nrst");
   strcpy(I_EFnames[50].funcname, "samplexz");
   strcpy(I_EFnames[51].funcname, "sampleyz");
   strcpy(I_EFnames[52].funcname, "scat2ddups");
   strcpy(I_EFnames[53].funcname, "scat2grid_bin_xy");
   strcpy(I_EFnames[54].funcname, "scat2grid_t");
   strcpy(I_EFnames[55].funcname, "scat2gridgauss_xy");
   strcpy(I_EFnames[56].funcname, "scat2gridgauss_xz");
   strcpy(I_EFnames[57].funcname, "scat2gridgauss_yz");
   strcpy(I_EFnames[58].funcname, "scat2gridgauss_xt");
   strcpy(I_EFnames[59].funcname, "scat2gridgauss_yt");
   strcpy(I_EFnames[60].funcname, "scat2gridgauss_zt");
   strcpy(I_EFnames[61].funcname, "scat2gridlaplace_xy");
   strcpy(I_EFnames[62].funcname, "scat2gridlaplace_xz");
   strcpy(I_EFnames[63].funcname, "scat2gridlaplace_yz");
   strcpy(I_EFnames[64].funcname, "scat2gridlaplace_xt");
   strcpy(I_EFnames[65].funcname, "scat2gridlaplace_yt");
   strcpy(I_EFnames[66].funcname, "scat2gridlaplace_zt");
   strcpy(I_EFnames[67].funcname, "scatgrid_nobs_xy");
   strcpy(I_EFnames[68].funcname, "sorti");
   strcpy(I_EFnames[69].funcname, "sortj");
   strcpy(I_EFnames[70].funcname, "sortk");
   strcpy(I_EFnames[71].funcname, "sortl");
   strcpy(I_EFnames[72].funcname, "tauto_cor");
   strcpy(I_EFnames[73].funcname, "tax_datestring");
   strcpy(I_EFnames[74].funcname, "tax_day");
   strcpy(I_EFnames[75].funcname, "tax_dayfrac");
   strcpy(I_EFnames[76].funcname, "tax_jday1900");
   strcpy(I_EFnames[77].funcname, "tax_jday");
   strcpy(I_EFnames[78].funcname, "tax_month");
   strcpy(I_EFnames[79].funcname, "tax_times");
   strcpy(I_EFnames[80].funcname, "tax_tstep");
   strcpy(I_EFnames[81].funcname, "tax_units");
   strcpy(I_EFnames[82].funcname, "tax_year");
   strcpy(I_EFnames[83].funcname, "tax_yearfrac");
   strcpy(I_EFnames[84].funcname, "tcat");
   strcpy(I_EFnames[85].funcname, "test_opendap");
   strcpy(I_EFnames[86].funcname, "treverse");
   strcpy(I_EFnames[87].funcname, "transpose_xt");
   strcpy(I_EFnames[88].funcname, "transpose_xy");
   strcpy(I_EFnames[89].funcname, "transpose_xz");
   strcpy(I_EFnames[90].funcname, "transpose_yt");
   strcpy(I_EFnames[91].funcname, "transpose_yz");
   strcpy(I_EFnames[92].funcname, "transpose_zt");
   strcpy(I_EFnames[93].funcname, "xcat");
   strcpy(I_EFnames[94].funcname, "xreverse");
   strcpy(I_EFnames[95].funcname, "ycat");
   strcpy(I_EFnames[96].funcname, "yreverse");
   strcpy(I_EFnames[97].funcname, "xauto_cor");
   strcpy(I_EFnames[98].funcname, "zaxreplace_avg");
   strcpy(I_EFnames[99].funcname, "zaxreplace_bin");
   strcpy(I_EFnames[100].funcname, "zaxreplace_rev");
   strcpy(I_EFnames[101].funcname, "zaxreplace_zlev");
   strcpy(I_EFnames[102].funcname, "zcat");
   strcpy(I_EFnames[103].funcname, "zreverse");
/*    
 *  ------------------------------------ 
 */



  if ( I_have_scanned_already ) {
    return_val = list_size(GLOBAL_ExternalFunctionList);
    return return_val;
  }

  if ( (GLOBAL_ExternalFunctionList = list_init()) == NULL ) {
    fprintf(stderr, "ERROR: efcn_scan: Unable to initialize GLOBAL_ExternalFunctionList.\n");
    return_val = -1;
    return return_val;
  }


  /*
   * Get internally linked external functions;  and add all 
   * the names and associated directory information to the 
   * GLOBAL_ExternalFunctionList.
   */


  /*
   * Read a name at a time.
   */

      for (i_intEF = 0; i_intEF < N_INTEF;   i_intEF = i_intEF + 1 ) {
	      strcpy(ef.path, "internally_linked");
	      strcpy(ef.name, I_EFnames[i_intEF].funcname);
	      ef.id = *gfcn_num_internal + ++count; /* pre-increment because F arrays start at 1 */
	      ef.already_have_internals = NO;
	      ef.internals_ptr = NULL;
	      list_insert_after(GLOBAL_ExternalFunctionList, &ef, sizeof(ExternalFunction));

      }

  /*
   * - Get all the paths from the "FER_EXTERNAL_FUNCTIONS" environment variable.
   *
   * - While there is another path:
   *    - get the path;
   *    - create a pipe for the "ls -1" command;
   *    - read stdout and use each file name to create another external function entry;
   *
   */

  if ( !getenv("FER_EXTERNAL_FUNCTIONS") ) {
    if ( !I_have_warned_already ) {
      fprintf(stderr, "\
\nWARNING: environment variable FER_EXTERNAL_FUNCTIONS not defined.\n\n");
      I_have_warned_already = TRUE;
    }
    /* *kob* v5.32 - the return val was set to 0 below but that was wrong. 
       That didn't take into account that on any system, the 
       FER_EXTERNAL_FUNCTIONS env variable might not be set.  If that were the
       case, a core dump occurred on all systems.  Set return_val to count, 
       which was generated above - also have to  note that the ef's 
       have been scanned*/
    return_val = count; 
    I_have_scanned_already = TRUE;
    return return_val;
  }

  sprintf(paths, "%s", getenv("FER_EXTERNAL_FUNCTIONS"));
    
  path_ptr = strtok(paths, " \t");

  if ( path_ptr == NULL ) {
 
    fprintf(stderr, "\
\nWARNING:No paths were found in the environment variable FER_EXTERNAL_FUNCTIONS.\n\n");

    return_val = 0;
    return return_val;
 
  } else {
    
    do {

	  strcpy(path, path_ptr);

      if (path[strlen(path)-1] != '/')
        strcat(path, "/"); 

      sprintf(cmd, "ls -1 %s", path);

      /* Open a pipe to the "ls" command */
      if ( (file_ptr = popen(cmd, "r")) == (FILE *) NULL ) {
	    fprintf(stderr, "\nERROR: Cannot open pipe.\n\n");
	    return_val = -1;
	    return return_val;
      }
 
      /*
       * Read a line at a time.
       * Any ~.so files are assumed to be external functions.
       */
      while ( fgets(file, EF_MAX_NAME_LENGTH, file_ptr) != NULL ) {

        char *extension;

	    file[strlen(file)-1] = '\0';   /* chop off the carriage return */
	    extension = &file[strlen(file)-3];
	    if ( strcmp(extension, ".so") == 0 ) {
          file[strlen(file)-3] = '\0'; /* chop off the ".so" */
	      strcpy(ef.path, path);
	      strcpy(ef.name, file);
	      ef.id = *gfcn_num_internal + ++count; /* pre-increment because F arrays start at 1 */
	      ef.already_have_internals = NO;
	      ef.internals_ptr = NULL;
	      list_insert_after(GLOBAL_ExternalFunctionList, &ef, sizeof(ExternalFunction));
	    }

      }
 
      pclose(file_ptr);
 
      path_ptr = strtok(NULL, " \t"); /* get the next directory */
 
    } while ( path_ptr != NULL );

    I_have_scanned_already = TRUE;
  }

  return_val = count;
  return return_val;

}


/*
 * Determine whether an external function has already 
 * had its internals read.
 */
int FORTRAN(efcn_already_have_internals)( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;

  static int return_val=0; /* static because it needs to exist after the return statement */


  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return return_val; }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

  return_val = ef_ptr->already_have_internals;

  return return_val;
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
  ExternalFunction *ef_ptr=NULL;
  ExternalFunctionInternals *i_ptr=NULL;
  int i=0, j=0;
  char ef_object[1024]="", tempText[EF_MAX_NAME_LENGTH]="", *c;
  int internally_linked = FALSE;

  static int return_val=0; /* static because it needs to exist after the return statement */

  
  void *handle;
  void (*f_init_ptr)(int *);

/* internal_dlsym() is declared to accept a char pointer (aka string)
 * and return a pointer to void (aka function pointer who's return type
 * is void).
 */
void *internal_dlsym(char *);


  /*
   * Find the external function.
   */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return return_val; }

  if (ef_ptr->already_have_internals)  { return return_val; }

  if ( (!strcmp(ef_ptr->path,"internally_linked")) ) {internally_linked = TRUE; }

  /*
   * Get a handle for the shared object.
   */
  if (!internally_linked) {
  strcat(ef_object, ef_ptr->path);
  strcat(ef_object, ef_ptr->name);
  strcat(ef_object, ".so");

     if ( (ef_ptr->handle = dlopen(ef_object, RTLD_LAZY)) == NULL ) {
       fprintf(stderr, "\n\
   ERROR in External Function %s:\n\
   Dynamic linking call dlopen() returns --\n\
   \"%s\".\n", ef_ptr->name, dlerror());
       return -1;
     }
  }  /* not internally_linked  */

  
  /*
   * Allocate and default initialize the internal information.
   * If anything went wrong, return the return_val.
   */

  return_val = EF_New(ef_ptr);

  if ( return_val != 0) {
    return return_val;
  }

  /*
   * Call the external function to really initialize the internal information.
   */
  i_ptr = ef_ptr->internals_ptr;

  if ( i_ptr->language == EF_C ) {

    fprintf(stderr, "\nERROR: C is not a supported language for External Functions.\n\n");
    return_val = -1;
    return return_val;

  } else if ( i_ptr->language == EF_F ) {


    /*
     * Prepare for bailout possibilities by setting a signal handler for
     * SIGFPE, SIGSEGV, SIGINT and SIGBUS and then by cacheing the stack 
     * environment with sigsetjmp (for the signal handler) and setjmp 
     * (for the "bail out" utility function).
     */   

    if ( EF_Util_setsig("efcn_gather_info")) {
       return;
    }

    /*
     * Set the signal return location and process jumps
     */
    if (sigsetjmp(sigjumpbuffer, 1) != 0) {
      return;
    }

    /*
     * Set the bail out return location and process jumps
     */
    if (setjmp(jumpbuffer) != 0) {
      return;
    }
    
    canjump = 1;

    /* Information about the overall function */


      sprintf(tempText, "");
      strcat(tempText, ef_ptr->name);
      strcat(tempText, "_init_");

      if (!internally_linked) {
         f_init_ptr = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
      } else {
        f_init_ptr = (void (*)(int *))internal_dlsym(tempText);
      }

      if (f_init_ptr == NULL) {
        fprintf(stderr, "ERROR in efcn_gather_info(): %s is not found.\n", tempText);
        fprintf(stderr, "  dlerror: %s\n", dlerror());
        return -1;
    }

    (*f_init_ptr)(id_ptr);

    ef_ptr->already_have_internals = TRUE;

    /*
     * Restore the old signal handlers.
     */
    if ( EF_Util_ressig("efcn_gather_info")) {
       return;
    }

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
  void *internal_dlsym(char *);

  /*
   * Initialize the status
   */
  *status = FERR_OK;

  /*
   * Store the context list globally.
   */
  EF_store_globals(NULL, NULL, cx_list_ptr, NULL, NULL);

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

    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
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

  } else {

    fprintf(stderr, "\nExternal Functions in C are not supported yet.\n\n");

  }

  return;
}


/*
 * Find an external function based on its integer ID, 
 * Query the function about abstract axes. Pass memory,
 * mr_list and cx_list info into the external function.
 */
void FORTRAN(efcn_get_result_limits)( int *id_ptr, float *memory, int *mr_list_ptr, int *cx_list_ptr, int *status )
{
  ExternalFunction *ef_ptr=NULL;
  char tempText[EF_MAX_NAME_LENGTH]="";
  int internally_linked = FALSE;

  void (*fptr)(int *);
  void *internal_dlsym(char *);

  /*
   * Initialize the status
   */
  *status = FERR_OK;

  /*
   * Store the memory pointer and various lists globally.
   */
  EF_store_globals(memory, mr_list_ptr, cx_list_ptr, NULL, NULL);

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


    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
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

  } else {

    fprintf(stderr, "\nExternal Functions in C are not supported yet.\n\n");

  }

  return;
}


/*
 * Find an external function based on its integer ID, 
 * pass the necessary information and the data and tell
 * the function to calculate the result.
 */
void FORTRAN(efcn_compute)( int *id_ptr, int *narg_ptr, int *cx_list_ptr, int *mr_list_ptr, int *mres_ptr,
	float *bad_flag_ptr, int *mr_arg_offset_ptr, float *memory, int *status )
{
  ExternalFunction *ef_ptr=NULL;
  ExternalFunctionInternals *i_ptr=NULL;
  float *arg_ptr[EF_MAX_COMPUTE_ARGS];
  int xyzt=0, i=0, j=0;
  int size=0;
  char tempText[EF_MAX_NAME_LENGTH]="";
  int internally_linked = FALSE;
  void *internal_dlsym(char *);


  /*
   * Prototype all the functions needed for varying numbers of
   * arguments and work arrays.
   */

  void (*fptr)(int *);
  void (*f1arg)(int *, float *, float *);
  void (*f2arg)(int *, float *, float *, float *);
  void (*f3arg)(int *, float *, float *, float *, float *);
  void (*f4arg)(int *, float *, float *, float *, float *, float *);
  void (*f5arg)(int *, float *, float *, float *, float *, float *, float *);
  void (*f6arg)(int *, float *, float *, float *, float *, float *, float *,
		float *);
  void (*f7arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *);
  void (*f8arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *);
  void (*f9arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *);
  void (*f10arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *);
  void (*f11arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *);
  void (*f12arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *);
  void (*f13arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *);
  void (*f14arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *);
  void (*f15arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *);
  void (*f16arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *);
  void (*f17arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *);
  void (*f18arg)(int *, float *, float *, float *, float *, float *, float *,
		float *, float *, float *, float *, float *, float *, float *, float *,
        float *, float *, float *, float *, float *);

  /*
   * Initialize the status
   */
  *status = FERR_OK;

  /*
   * Store the array dimensions for memory resident variables and for working storage.
   * Store the memory pointer and various lists globally.
   */
  FORTRAN(efcn_copy_array_dims)();
  EF_store_globals(memory, mr_list_ptr, cx_list_ptr, mres_ptr, bad_flag_ptr);

  /*
   * Find the external function.
   */
  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) {
    fprintf(stderr, "\n\
ERROR in efcn_compute() finding external function: id = [%d]\n", *id_ptr);
    *status = FERR_EF_ERROR;
    return;
  }
  if ( (!strcmp(ef_ptr->path,"internally_linked")) ) {internally_linked = TRUE; }

  i_ptr = ef_ptr->internals_ptr;

  if ( i_ptr->language == EF_F ) {

    /*
     * Begin assigning the arg_ptrs.
     */

    /* First come the arguments to the function. */

     for (i=0; i<i_ptr->num_reqd_args; i++) {
       arg_ptr[i] = memory + mr_arg_offset_ptr[i];
     }

    /* Now for the result */

     arg_ptr[i++] = memory + mr_arg_offset_ptr[EF_MAX_ARGS];

    /* Now for the work arrays */

    /*
     * If this program has requested working storage we need to 
     * ask the function to specify the amount of space needed
     * and then create the memory here.  Memory will be released
     * after the external function returns.
     */
    if (i_ptr->num_work_arrays > EF_MAX_WORK_ARRAYS) {

	  fprintf(stderr, "\n\
ERROR specifying number of work arrays in ~_init subroutine of external function %s\n\
\tnum_work_arrays[=%d] exceeds maximum[=%d].\n\n", ef_ptr->name, i_ptr->num_work_arrays, EF_MAX_WORK_ARRAYS);
	  *status = FERR_EF_ERROR;
	  return;

    } else if (i_ptr->num_work_arrays < 0) {

	  fprintf(stderr, "\n\
ERROR specifying number of work arrays in ~_init subroutine of external function %s\n\
\tnum_work_arrays[=%d] must be a positive number.\n\n", ef_ptr->name, i_ptr->num_work_arrays);
	  *status = FERR_EF_ERROR;
	  return;

    } else if (i_ptr->num_work_arrays > 0)  {

      sprintf(tempText, "");
      strcat(tempText, ef_ptr->name);
      strcat(tempText, "_work_size_");

      if (!internally_linked) {
         fptr = (void (*)(int *))dlsym(ef_ptr->handle, tempText);
      } else {
         fptr  = (void (*)(int *))internal_dlsym(tempText);
      }

      if (fptr == NULL) {
	fprintf(stderr, "\n\
ERROR in efcn_compute() accessing %s\n", tempText);
	*status = FERR_EF_ERROR;
        return;
      }
      (*fptr)( id_ptr );


	  /* Allocate memory for each individual work array */

      for (j=0; j<i_ptr->num_work_arrays; i++, j++) {

        int iarray,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi;
        iarray = j+1;
        xlo = i_ptr->work_array_lo[j][0];
        ylo = i_ptr->work_array_lo[j][1];
        zlo = i_ptr->work_array_lo[j][2];
        tlo = i_ptr->work_array_lo[j][3];
        xhi = i_ptr->work_array_hi[j][0];
        yhi = i_ptr->work_array_hi[j][1];
        zhi = i_ptr->work_array_hi[j][2];
        thi = i_ptr->work_array_hi[j][3];

        FORTRAN(efcn_set_work_array_dims)(&iarray,&xlo,&ylo,&zlo,&tlo,&xhi,&yhi,&zhi,&thi);

        size = sizeof(float) * (xhi-xlo+1) * (yhi-ylo+1) * (zhi-zlo+1) * (thi-tlo+1);

        if ( (arg_ptr[i] = (float *)malloc(size)) == NULL ) { 
          fprintf(stderr, "\n\
ERROR in efcn_compute() allocating %d bytes of memory\n\
      work array %d:  X=%d:%d, Y=%d:%d, Z=%d:%d, T=%d:%d\n", 
      size,iarray,xlo,xhi,ylo,yhi,zlo,zhi,tlo,thi);
	  *status = FERR_EF_ERROR;
	  return;
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


    /*
     * Now go ahead and call the external function's "_compute_" function,
     * prototyping it for the number of arguments expected.
     */
    sprintf(tempText, "");
    strcat(tempText, ef_ptr->name);
    strcat(tempText, "_compute_");

    switch ( i_ptr->num_reqd_args + i_ptr->num_work_arrays ) {

    case 1:
	  if (!internally_linked) {
            f1arg  = (void (*)(int *, float *, float *))
             dlsym(ef_ptr->handle, tempText);
          } else {
	    f1arg  = (void (*)(int *, float *, float *))
             internal_dlsym(tempText);
          }
	  (*f1arg)( id_ptr, arg_ptr[0], arg_ptr[1] );
	break;


    case 2:
	  if (!internally_linked) {
            f2arg  = (void (*)(int *, float *, float *, float *))
             dlsym(ef_ptr->handle, tempText);
          } else {
            f2arg  = (void (*)(int *, float *, float *, float *))
             internal_dlsym(tempText);
          }
	  (*f2arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2] );
	break;


    case 3:
	  if (!internally_linked) {
	     f3arg  = (void (*)(int *, float *, float *, float *, float *))
              dlsym(ef_ptr->handle, tempText);
          } else {
	     f3arg  = (void (*)(int *, float *, float *, float *, float *))
              internal_dlsym(tempText);
          }
	  (*f3arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3] );
	break;


    case 4:
	  if (!internally_linked) {
            f4arg  = (void (*)(int *, float *, float *, float *, float *, float *))
             dlsym(ef_ptr->handle, tempText);
          } else {
            f4arg  = (void (*)(int *, float *, float *, float *, float *, float *))
             internal_dlsym(tempText);
          }
	  (*f4arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4] );
	break;


    case 5:
	  if (!internally_linked) {
	    f5arg  = (void (*)(int *, float *, float *, float *, float *, float *, 
             float *))
             dlsym(ef_ptr->handle, tempText);
          } else {
	    f5arg  = (void (*)(int *, float *, float *, float *, float *, float *, 
             float *))
             internal_dlsym(tempText);
          }
	  (*f5arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5] );
	break;


    case 6:
	  if (!internally_linked) {
	    f6arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f6arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *))internal_dlsym(tempText);
          }
	  (*f6arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6] );
	break;


    case 7:
	  if (!internally_linked) {
	    f7arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f7arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *))internal_dlsym(tempText);
          }
	  (*f7arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7] );
	break;


    case 8:
	  if (!internally_linked) {
	    f8arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f8arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *))internal_dlsym(tempText);
          }
	  (*f8arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8] );
	break;


    case 9:
	  if (!internally_linked) {
            f9arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
          } else {
            f9arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *))internal_dlsym(tempText);
          }
	  (*f9arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9] );
	break;


    case 10:
	  if (!internally_linked) {
	    f10arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f10arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *))internal_dlsym(tempText);
          }
	  (*f10arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10] );
	break;


    case 11:
	  if (!internally_linked) {
            f11arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *))
             dlsym(ef_ptr->handle, tempText);
          } else {
            f11arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *))
             internal_dlsym(tempText);
          }
	  (*f11arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11] );
	break;


    case 12:
	  if (!internally_linked) {
	    f12arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *))
             dlsym(ef_ptr->handle, tempText);
          } else {
	    f12arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *))
             internal_dlsym(tempText);
          }
	  (*f12arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12] );
	break;


    case 13:
	  if (!internally_linked) {
	    f13arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *))
             dlsym(ef_ptr->handle, tempText);
          } else {
	    f13arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *))
             internal_dlsym(tempText);
          }
	  (*f13arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13] );
	break;


    case 14:
	  if (!internally_linked) {
	    f14arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *,
             float *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f14arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *,
             float *))internal_dlsym(tempText);
          }
	  (*f14arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14] );
	break;


    case 15:
	  if (!internally_linked) {
	   f15arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
            float *, float *, float *, float *, float *, float *, float *, float *,
            float *, float *))dlsym(ef_ptr->handle, tempText);
          } else {
	   f15arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
            float *, float *, float *, float *, float *, float *, float *, float *,
            float *, float *))internal_dlsym(tempText);
          }
	  (*f15arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15] );
	break;


    case 16:
	  if (!internally_linked) {
	    f16arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f16arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *))internal_dlsym(tempText);
          }
	  (*f16arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15], arg_ptr[16] );
	break;


    case 17:
	  if (!internally_linked) {
            f17arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
          } else {
            f17arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *))internal_dlsym(tempText);
          }
	  (*f17arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15], arg_ptr[16],
        arg_ptr[17] );
	break;


    case 18:
	  if (!internally_linked) {
	    f18arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *))dlsym(ef_ptr->handle, tempText);
          } else {
	    f18arg  = (void (*)(int *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *, float *, float *, float *,
             float *, float *, float *, float *, float *))internal_dlsym(tempText);
          }
	  (*f18arg)( id_ptr, arg_ptr[0], arg_ptr[1], arg_ptr[2], arg_ptr[3], arg_ptr[4],
        arg_ptr[5], arg_ptr[6], arg_ptr[7], arg_ptr[8], arg_ptr[9], arg_ptr[10],
        arg_ptr[11], arg_ptr[12], arg_ptr[13], arg_ptr[14], arg_ptr[15], arg_ptr[16],
        arg_ptr[17], arg_ptr[18] );
	break;


    default:
      fprintf(stderr, "\n\
ERROR: External functions with more than %d arguments are not implemented yet.\n\n", EF_MAX_ARGS);
      *status = FERR_EF_ERROR;
      return;
      break;

    }

      /*
       * Restore the old signal handlers.
       */
    if ( EF_Util_ressig("efcn_compute")) {
       return;
    }


    /*
     * Now it's time to release the work space.
     * With arg_ptr[0] for argument #1, and remembering one slot for the result,
     * we should begin freeing up memory at arg_ptr[num_reqd_args+1].
     */
    for (i=i_ptr->num_reqd_args+1; i<i_ptr->num_reqd_args+1+i_ptr->num_work_arrays; i++) {
      free(arg_ptr[i]);
    }

  } else if ( ef_ptr->internals_ptr->language == EF_C ) {

    fprintf(stderr, "\n\
ERROR: External Functions may not yet be written in C.\n\n");
    *status = FERR_EF_ERROR;
    return;

  }
  
  return;
}


/*
 * A signal handler for SIGFPE, SIGSEGV, SIGINT and SIGBUS signals generated
 * while executing an external function.  See "Advanced Programming
 * in the UNIX Environment" p. 299 ff for details.
 */
static void EF_signal_handler(int signo) {

  if (canjump == 0) return; /* unexpected signal, ignore */

  /*
      /*
       * Restore the old signal handlers.
       */
    if ( EF_Util_ressig("efcn_compute")) {
       return;
    }

  if (signo == SIGFPE) {
    fprintf(stderr, "\n\nERROR in external function: Floating Point Error\n");
    canjump = 0;
    siglongjmp(sigjumpbuffer, 1);
  } else if (signo == SIGSEGV) {
    fprintf(stderr, "\n\nERROR in external function: Segmentation Violation\n");
    canjump = 0;
    siglongjmp(sigjumpbuffer, 1);
  } else if (signo == SIGINT) {
    fprintf(stderr, "\n\nExternal function halted with Control-C\n");
    canjump = 0;
    siglongjmp(sigjumpbuffer, 1);
  } else if (signo == SIGBUS) {
    fprintf(stderr, "\n\nERROR in external function: Hardware Fault\n");
    canjump = 0;
    siglongjmp(sigjumpbuffer, 1);
  } else {
    fprintf(stderr, "\n\nERROR in external function: signo = %d\n", signo);
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

  status = list_traverse(GLOBAL_ExternalFunctionList, name, EF_ListTraverse_FoundName, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, set the id_ptr to ATOM_NOT_FOUND.
   */
  if ( status != LIST_OK ) {
    return_val = ATOM_NOT_FOUND;
    return return_val;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 

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
  int status=LIST_OK;
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
void FORTRAN(efcn_get_version)( int *id_ptr, float *version )
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
 * return the number of arguments.
 */
int FORTRAN(efcn_get_num_reqd_args)( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;

  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

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
 * float or a string.
 */
int FORTRAN(efcn_get_arg_type)( int *id_ptr, int *iarg_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  static int return_val=0; /* static because it needs to exist after the return statement */
  int index = *iarg_ptr - 1; /* C indices are 1 less than Fortran */ 

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
  return_val = ef_ptr->internals_ptr->arg_type[index];
  
  return return_val;
}


/*
 * Find an external function based on its integer ID and
 * return the 'rtn_type' information for the result which
 * tells Ferret whether an argument is a float or a string.
 */
int FORTRAN(efcn_get_rtn_type)( int *id_ptr )
{
  ExternalFunction *ef_ptr=NULL;
  static int return_val=0; /* static because it needs to exist after the return statement */

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }
  
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
  
  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
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



void FORTRAN(ef_err_bail_out)(int *id_ptr, char *text)
{
  ExternalFunction *ef_ptr=NULL;

  if ( (ef_ptr = ef_ptr_from_id_ptr(id_ptr)) == NULL ) { return; }

  fprintf(stderr, "\n\
Bailing out of external function \"%s\":\n\
\t%s\n", ef_ptr->name, text);

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
  int status=LIST_OK, i=0, j=0;

  static int return_val=0; /* static because it needs to exist after the return statement */


  /*
   * Allocate space for the internals.
   * If the allocation failed, print a warning message and return.
   */

  this->internals_ptr = malloc(sizeof(ExternalFunctionInternals));
  i_ptr = this->internals_ptr;

  if ( i_ptr == NULL ) {
    fprintf(stderr, "ERROR in EF_New(): cannot allocate ExternalFunctionInternals.\n");
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
  for (i=0; i<4; i++) {
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
    for (j=0; j<4; j++) {
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


/* .... UtilityFunctions for dealing with GLOBAL_ExternalFunctionList .... */

/*
 * Store the global values which will be needed by utility routines
 * in EF_ExternalUtil.c
 */
void EF_store_globals(float *memory_ptr, int *mr_list_ptr, int *cx_list_ptr, 
	int *mres_ptr, float *bad_flag_ptr)
{
  int i=0;

  GLOBAL_memory_ptr = memory_ptr;
  GLOBAL_mr_list_ptr = mr_list_ptr;
  GLOBAL_cx_list_ptr = cx_list_ptr;
  GLOBAL_mres_ptr = mres_ptr;
  GLOBAL_bad_flag_ptr = bad_flag_ptr;

}


/*
 * Find an external function based on an integer id and return
 * the ef_ptr.
 */
ExternalFunction *ef_ptr_from_id_ptr(int *id_ptr)
{
  static ExternalFunction *ef_ptr=NULL;
  int status=LIST_OK;

  status = list_traverse(GLOBAL_ExternalFunctionList, id_ptr, EF_ListTraverse_FoundID, (LIST_FRNT | LIST_FORW | LIST_ALTR));

  /*
   * If the search failed, print a warning message and return.
   */
  if ( status != LIST_OK ) {
    fprintf(stderr, "\nERROR: in ef_ptr_from_id_ptr: No external function of id %d was found.\n\n", *id_ptr);
    return NULL;
  }

  ef_ptr=(ExternalFunction *)list_curr(GLOBAL_ExternalFunctionList); 
  
  return ef_ptr;
}


int EF_ListTraverse_fprintf( char *data, char *curr )
{
  FILE *File_ptr=(FILE *)data;
  ExternalFunction *ef_ptr=(ExternalFunction *)curr; 
     
  fprintf(stderr, "path = \"%s\", name = \"%s\", id = %d, internals_ptr = %d\n",
	  ef_ptr->path, ef_ptr->name, ef_ptr->id, ef_ptr->internals_ptr);

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
	if ( ++n == '\0' ) /* end of name */
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
      fprintf(stderr, "\nERROR in %s() catching SIGFPE.\n", fcn_name);
      return 1;
    }
    if ( (segv_handler = signal(SIGSEGV, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in %s() catching SIGSEGV.\n", fcn_name);
      return 1;
    }
    if ( (int_handler = signal(SIGINT, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in %s() catching SIGINT.\n", fcn_name);
      return 1;
    }
    if ( (bus_handler = signal(SIGBUS, EF_signal_handler)) == SIG_ERR ) {
      fprintf(stderr, "\nERROR in %s() catching SIGBUS.\n", fcn_name);
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
      fprintf(stderr, "\nERROR in %s() restoring default SIGFPE handler.\n", fcn_name);
      return 1;
    }
    if (signal(SIGSEGV, (*segv_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in %s() restoring default SIGSEGV handler.\n", fcn_name);
      return 1;
    }
    if (signal(SIGINT, (*int_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in %s() restoring default SIGINT handler.\n", fcn_name);
      return 1;
    }
    if (signal(SIGBUS, (*bus_handler)) == SIG_ERR) {
      fprintf(stderr, "\nERROR in %s() restoring default SIGBUS handler.\n", fcn_name);
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

void *internal_dlsym(char *name) {

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
else if ( !strcmp(name,"sampleij_init_") ) return (void *)sampleij_init_;
else if ( !strcmp(name,"sampleij_result_limits_") ) return (void *)sampleij_result_limits_;
else if ( !strcmp(name,"sampleij_work_size_") ) return (void *)sampleij_work_size_;
else if ( !strcmp(name,"sampleij_compute_") ) return (void *)sampleij_compute_;

/* samplet_date.F */
else if ( !strcmp(name,"samplet_date_init_") ) return (void *)samplet_date_init_;
else if ( !strcmp(name,"samplet_date_result_limits_") ) return (void *)samplet_date_result_limits_;
else if ( !strcmp(name,"samplet_date_work_size_") ) return (void *)samplet_date_work_size_;
else if ( !strcmp(name,"samplet_date_compute_") ) return (void *)samplet_date_compute_;

/* samplexy.F */
else if ( !strcmp(name,"samplexy_init_") ) return (void *)samplexy_init_;
else if ( !strcmp(name,"samplexy_result_limits_") ) return (void *)samplexy_result_limits_;
else if ( !strcmp(name,"samplexy_work_size_") ) return (void *)samplexy_work_size_;
else if ( !strcmp(name,"samplexy_compute_") ) return (void *)samplexy_compute_;

/* samplexy_curv.F */
else if ( !strcmp(name,"samplexy_curv_init_") ) return (void *)samplexy_curv_init_;
else if ( !strcmp(name,"samplexy_curv_result_limits_") ) return (void *)samplexy_curv_result_limits_;
else if ( !strcmp(name,"samplexy_curv_work_size_") ) return (void *)samplexy_curv_work_size_;
else if ( !strcmp(name,"samplexy_curv_compute_") ) return (void *)samplexy_curv_compute_;

/* samplexy_curv_avg.F */
else if ( !strcmp(name,"samplexy_curv_avg_init_") ) return (void *)samplexy_curv_avg_init_;
else if ( !strcmp(name,"samplexy_curv_avg_result_limits_") ) return (void *)samplexy_curv_avg_result_limits_;
else if ( !strcmp(name,"samplexy_curv_avg_work_size_") ) return (void *)samplexy_curv_avg_work_size_;
else if ( !strcmp(name,"samplexy_curv_avg_compute_") ) return (void *)samplexy_curv_avg_compute_;

/* samplexy_curv_nrst.F */
else if ( !strcmp(name,"samplexy_curv_nrst_init_") ) return (void *)samplexy_curv_nrst_init_;
else if ( !strcmp(name,"samplexy_curv_nrst_result_limits_") ) return (void *)samplexy_curv_nrst_result_limits_;
else if ( !strcmp(name,"samplexy_curv_nrst_work_size_") ) return (void *)samplexy_curv_nrst_work_size_;
else if ( !strcmp(name,"samplexy_curv_nrst_compute_") ) return (void *)samplexy_curv_nrst_compute_;

/* samplexy_closest.F */
else if ( !strcmp(name,"samplexy_closest_init_") ) return (void *)samplexy_closest_init_;
else if ( !strcmp(name,"samplexy_closest_result_limits_") ) return (void *)samplexy_closest_result_limits_;
else if ( !strcmp(name,"samplexy_closest_work_size_") ) return (void *)samplexy_closest_work_size_;
else if ( !strcmp(name,"samplexy_closest_compute_") ) return (void *)samplexy_closest_compute_;

/* samplexz.F */
else if ( !strcmp(name,"samplexz_init_") ) return (void *)samplexz_init_;
else if ( !strcmp(name,"samplexz_result_limits_") ) return (void *)samplexz_result_limits_;
else if ( !strcmp(name,"samplexz_work_size_") ) return (void *)samplexz_work_size_;
else if ( !strcmp(name,"samplexz_compute_") ) return (void *)samplexz_compute_;

/* sampleyz.F */
else if ( !strcmp(name,"sampleyz_init_") ) return (void *)sampleyz_init_;
else if ( !strcmp(name,"sampleyz_result_limits_") ) return (void *)sampleyz_result_limits_;
else if ( !strcmp(name,"sampleyz_work_size_") ) return (void *)sampleyz_work_size_;
else if ( !strcmp(name,"sampleyz_compute_") ) return (void *)sampleyz_compute_;

/* scat2grid_bin_xy.F */
else if ( !strcmp(name,"scat2grid_bin_xy_init_") ) return (void *)FORTRAN(scat2grid_bin_xy_init);
else if ( !strcmp(name,"scat2grid_bin_xy_work_size_") ) return (void *)FORTRAN(scat2grid_bin_xy_work_size);
else if ( !strcmp(name,"scat2grid_bin_xy_compute_") ) return (void *)FORTRAN(scat2grid_bin_xy_compute);

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

/* scatgrid_nobs_xy.F */
else if ( !strcmp(name,"scatgrid_nobs_xy_init_") ) return (void *)FORTRAN(scatgrid_nobs_xy_init);
else if ( !strcmp(name,"scatgrid_nobs_xy_work_size_") ) return (void *)FORTRAN(scatgrid_nobs_xy_work_size);
else if ( !strcmp(name,"scatgrid_nobs_xy_compute_") ) return (void *)FORTRAN(scatgrid_nobs_xy_compute);

/* sorti.F */
else if ( !strcmp(name,"sorti_init_") ) return (void *)sorti_init_;
else if ( !strcmp(name,"sorti_result_limits_") ) return (void *)sorti_result_limits_;
else if ( !strcmp(name,"sorti_work_size_") ) return (void *)sorti_work_size_;
else if ( !strcmp(name,"sorti_compute_") ) return (void *)sorti_compute_;

/* sortj.F */
else if ( !strcmp(name,"sortj_init_") ) return (void *)sortj_init_;
else if ( !strcmp(name,"sortj_result_limits_") ) return (void *)sortj_result_limits_;
else if ( !strcmp(name,"sortj_work_size_") ) return (void *)sortj_work_size_;
else if ( !strcmp(name,"sortj_compute_") ) return (void *)sortj_compute_;

/* sortk.F */
else if ( !strcmp(name,"sortk_init_") ) return (void *)sortk_init_;
else if ( !strcmp(name,"sortk_result_limits_") ) return (void *)sortk_result_limits_;
else if ( !strcmp(name,"sortk_work_size_") ) return (void *)sortk_work_size_;
else if ( !strcmp(name,"sortk_compute_") ) return (void *)sortk_compute_;

/* sortl.F */
else if ( !strcmp(name,"sortl_init_") ) return (void *)sortl_init_;
else if ( !strcmp(name,"sortl_result_limits_") ) return (void *)sortl_result_limits_;
else if ( !strcmp(name,"sortl_work_size_") ) return (void *)sortl_work_size_;
else if ( !strcmp(name,"sortl_compute_") ) return (void *)sortl_compute_;

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

/* rect_to_curv.F */
else if ( !strcmp(name,"rect_to_curv_init_") ) return (void *)FORTRAN(rect_to_curv_init);
else if ( !strcmp(name,"rect_to_curv_work_size_") ) return (void *)FORTRAN(rect_to_curv_work_size);
else if ( !strcmp(name,"rect_to_curv_compute_") ) return (void *)FORTRAN(rect_to_curv_compute);

/* date1900.F */
else if ( !strcmp(name,"date1900_init_") ) return (void *)FORTRAN(date1900_init);
else if ( !strcmp(name,"date1900_result_limits_") ) return (void *)FORTRAN(date1900_result_limits);
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

/* transpose_yt.F */
else if ( !strcmp(name,"transpose_yt_init_") ) return (void *)FORTRAN(transpose_yt_init);
else if ( !strcmp(name,"transpose_yt_result_limits_") ) return (void *)FORTRAN(transpose_yt_result_limits);
else if ( !strcmp(name,"transpose_yt_compute_") ) return (void *)FORTRAN(transpose_yt_compute);

/* transpose_yz.F */
else if ( !strcmp(name,"transpose_yz_init_") ) return (void *)FORTRAN(transpose_yz_init);
else if ( !strcmp(name,"transpose_yz_result_limits_") ) return (void *)FORTRAN(transpose_yz_result_limits);
else if ( !strcmp(name,"transpose_yz_compute_") ) return (void *)FORTRAN(transpose_yz_compute);

/* transpose_zt.F */
else if ( !strcmp(name,"transpose_zt_init_") ) return (void *)FORTRAN(transpose_zt_init);
else if ( !strcmp(name,"transpose_zt_result_limits_") ) return (void *)FORTRAN(transpose_zt_result_limits);
else if ( !strcmp(name,"transpose_zt_compute_") ) return (void *)FORTRAN(transpose_zt_compute);

/* xcat.F */
else if ( !strcmp(name,"xcat_init_") ) return (void *)FORTRAN(xcat_init);
else if ( !strcmp(name,"xcat_result_limits_") ) return (void *)FORTRAN(xcat_result_limits);
else if ( !strcmp(name,"xcat_compute_") ) return (void *)FORTRAN(xcat_compute);

/* ycat.F */
else if ( !strcmp(name,"ycat_init_") ) return (void *)FORTRAN(ycat_init);
else if ( !strcmp(name,"ycat_result_limits_") ) return (void *)FORTRAN(ycat_result_limits);
else if ( !strcmp(name,"ycat_compute_") ) return (void *)FORTRAN(ycat_compute);

/* zcat.F */
else if ( !strcmp(name,"zcat_init_") ) return (void *)FORTRAN(zcat_init);
else if ( !strcmp(name,"zcat_result_limits_") ) return (void *)FORTRAN(zcat_result_limits);
else if ( !strcmp(name,"zcat_compute_") ) return (void *)FORTRAN(zcat_compute);

/* tcat.F */
else if ( !strcmp(name,"tcat_init_") ) return (void *)FORTRAN(tcat_init);
else if ( !strcmp(name,"tcat_result_limits_") ) return (void *)FORTRAN(tcat_result_limits);
else if ( !strcmp(name,"tcat_compute_") ) return (void *)FORTRAN(tcat_compute);

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
else if ( !strcmp(name,"tax_month_compute_") ) return (void *)FORTRAN(tax_month_compute);

else if ( !strcmp(name,"tax_times_init_") ) return (void *)FORTRAN(tax_times_init);
else if ( !strcmp(name,"tax_times_compute_") ) return (void *)FORTRAN(tax_times_compute);

else if ( !strcmp(name,"tax_tstep_init_") ) return (void *)FORTRAN(tax_tstep_init);
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

 }
/*  End of function pointer list for internally-linked External Functions
 *  ------------------------------------ */
