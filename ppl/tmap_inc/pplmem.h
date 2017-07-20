#ifndef _PPLMEM_H_
#define _PPLMEM_H_

/* pplmem.h 
   Declarations for routines that allow dynamic PPLUS memory buffer 
   9/18/01 *acm*

*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
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
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
*/

/* Better if these were defined in only one include file, but .... */
#ifndef FORTRAN
#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif
#endif

#ifndef DFTYPE
#ifdef double_p
#define DFTYPE double
#else
#define DFTYPE float
#endif
#endif

/* pointer to memory to be used by PPL - allocated by ferret */
extern float *ppl_memory;

/* now provided by libpyferret.c which just int instead of int* */
void reallo_ppl_memory(int this_size);

void FORTRAN(get_ppl_memory_size)(int *plot_mem_used);
int  FORTRAN(its_gksm)(int *wkid);
void FORTRAN(pplcmd_c)(int *isi, int *icmdim, int *icmsze);
void FORTRAN(pplcmd_f)(int *isi, int *icmdim, int *icmsze, float *plot_memory);
void FORTRAN(pplld_pts)(int *npts, float *plot_memory);
void FORTRAN(pplld_pts_envelope)(int *npts, int *plot_mem_used);
void FORTRAN(pplldc)(int *k, DFTYPE *z, int *mx, int *my,int *imn, int *imx,
                     int *jmn, int *jmx, DFTYPE *pi, DFTYPE *pj,int *nx1, int *ny1,
                     DFTYPE *xmin1, DFTYPE *ymin1, DFTYPE *dx1, DFTYPE *dy1, float *plot_mem_used);
void FORTRAN(pplldc_envelope)(int *k, DFTYPE *z, int *mx, int *my,int *imn, int *imx,
                              int *jmn, int *jmx, DFTYPE *pi, DFTYPE *pj,int *nx1, int *ny1,
                              DFTYPE *xmin1, DFTYPE *ymin1, DFTYPE *dx1, DFTYPE *dy1, int *plot_mem_used);
void FORTRAN(pplldv)(int *K, DFTYPE *Z, int *MX, int *MY, int *IMN,int *IMX, int *JMN, int *JMX, float *plot_memory);
void FORTRAN(pplldv_envelope)(int *K, DFTYPE *Z, int *MX, int *MY, int *IMN,int *IMX, int *JMN, int *JMX);
void FORTRAN(pplldx)(int *icode, DFTYPE *xt, DFTYPE *yt, int *npts, char *tstrt, char *tref, DFTYPE *xdt, float *plot_memory);
void FORTRAN(pplldx_envelope)(int *icode, DFTYPE *xt, DFTYPE *yt, int *npts, char *tstrt, char *tref, DFTYPE *xdt, int *plot_mem_used);
void FORTRAN(reallo_envelope)(int *plot_mem_used);
void FORTRAN(resize_xgks_window)(int *ws_id, float *x, float *y, int *ix, int *iy);
void FORTRAN(save_ppl_memory_size)(int *plot_mem_used);
void FORTRAN(set_background)(int *ws_id, int *ndx);
void FORTRAN(wait_on_resize)(int *ws_id);
void FORTRAN(xgks_x_events)(void);

#endif
