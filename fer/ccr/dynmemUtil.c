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

/* dynmemUtil.c
 *
 * Steve Hankin
 * Jan. 2017
 *
 * This file contains the utility functions which Ferret
 * needs in order to pass an array of dynamic memory pointers (mr_list)
 * to c.  It bypasses the difficulties of passing the FORTRAN derived type
 * of pointers directly as an array
 *
 * V702 *sh*   1/17 
*/



/* .................... Includes .................... */

#include <sys/types.h>	        /* required for some of our prototypes */
#include <sys/stat.h>
#include <sys/errno.h>

#include <unistd.h>		/* for convenience */
#include <stdlib.h>		/* for convenience */
#include <stdio.h>		/* for convenience */
#include <string.h>		/* for convenience */

#include "ferret.h"
#include "FerMem.h"
#include "EF_Util.h"


/*
 * Create a pointer array in c.  Pass the pointer back as an long int value.
 */
void FORTRAN(dynmem_make_ptr_array)(int* n, long* mr_ptrs_val, int* status)
{
  int size;
  DFTYPE** mr_ptrs;
  *status = FERR_OK;  // default

  size = sizeof(DFTYPE*) * *n;
  mr_ptrs = (DFTYPE**)FerMem_Malloc(size);

  if ( mr_ptrs == NULL ) { 
    fprintf(stderr, "**ERROR in dynmem_make_ptr_array");
    *status = FERR_EF_ERROR;
    return;
  }


  *mr_ptrs_val = (long)mr_ptrs; 
}

/*
 * Insert one pointer (from FORTRAN) into the c pointer array
 */
void FORTRAN(dynmem_pass_1_ptr)(int* iarg, DFTYPE* arg_ptr, long* mr_ptrs_val)
{
  int iarg_c = *iarg-1;   // FORTRAN index to c index
  DFTYPE**  mr_ptrs;

  mr_ptrs = (DFTYPE**) *mr_ptrs_val;

  mr_ptrs[iarg_c]  = arg_ptr;
}


/*
 * Free the pointer array
 */
void FORTRAN(dynmem_free_ptr_array)(long* mr_ptrs_val)
{
  DFTYPE** mr_ptrs;

  mr_ptrs = (DFTYPE**) *mr_ptrs_val;

  FerMem_Free(mr_ptrs);
}


