/* pplld_pts_envelope.c
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

   This intermediate "envelope" routine provides the global pointer
   to the PLOT+ memory buffer "X" (called ppl_memory here, X in plot routines). 
   If the current buffer size is too small this routine ensures that it 
   gets enlarged. The final act is  to call the PLOT+ PPLLD_PTS routine -- 
   the original arguments are simply passed through, with a new argument: 
   the PPLUS memory buffer.

*	subroutine pplld_pts(npts,xdt)
*/

/*******************/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "pplmem.h"
/*******************/

/* The global pointer to PLOT+ memory is declared as extern here
   (Defined in fermain_c.c)
*/
  extern float *ppl_memory; 
  
/* pplld_pts_envelope: this routine, called from FORTRAN, will check the
   memory available for plotting, allocate more if needed, and call pplld_pts.
 */

void FORTRAN(pplld_pts_envelope)(int *npts,int *plot_mem_used) 
{  
/* local variable declaration */
  int pmemsize;
/*
  Is the currently allocated size of PLOT+ memory sufficient?
  If not, then allocate a larger array
  Note need to check if the reallocation is successful.
*/

  FORTRAN(get_ppl_memory_size)(&pmemsize);

  if (*plot_mem_used > pmemsize) reallo_ppl_memory(plot_mem_used); 

  FORTRAN(pplld_pts) (npts, ppl_memory);

return;
}
