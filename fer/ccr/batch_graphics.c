/*
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
*/

#include <Python.h> /* make sure Python.h is first */
#include <assert.h>
#include <string.h>
#include "grdel.h"

#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

/* set_batch_graphics */
void FORTRAN(set_batch_graphics)(char *outfile)
{
  int length;
  int status;

  assert( outfile != NULL );
  length = strlen(outfile);
  /*
   * This can be called either with (-batch) or without
   * (-gif or -unmapped) a filename.  Only call
   * save_metafile_name if a filename is given (-batch).
   */
  if ( length > 0 ) {
     FORTRAN(save_metafile_name)(outfile, &length);
     FORTRAN(assign_modemeta)();
  }

  /* 
   * GKS metafile format no longer supported. The "-batch",
   * "-gif", and "-unmapped" options do not change the workflow.
   * If one of these option is given, however, windows are not
   * made visible.  This allows the use of a faster graphics engine.
   */
  FORTRAN(fgd_hide_all_windows)();
  /* This assume GCC standard for passing Holerith strings */
  FORTRAN(fgd_set_engine)("CAIRO", &status, 5);

  return;
}

