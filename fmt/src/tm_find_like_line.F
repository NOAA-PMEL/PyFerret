	INTEGER*4 FUNCTION TM_Find_like_line ( iline )
*
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
*
*
* Check to see if an existing line is identical to the line pointed to by
* "iline".
*
* by M. Verschell PMEL/TMAP
*
* revision 0.00 - 05/18/88
* revision 0.01 - 11/21/88 - variable used check changed to specific char_initXX
* revision 0.02 - 09/21/89 - *sh* irregular line coordinate comparison fixed
* revision 0.02 - 2/92 - *sh* removed guts of routine to TM_SAME_LINE_DEF

* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* Argument definitions
	INTEGER*4	iline

* Local definitions
        LOGICAL         TM_SAME_LINE_DEF
	INTEGER*4	i

* Loop through defined lines
	DO 100 i = 1, max_lines

	  IF (i.NE.iline .AND. line_name(i).NE.char_init16) THEN

             IF (TM_SAME_LINE_DEF( i, iline ) ) THEN
	        TM_Find_like_line = i
                RETURN
	     ENDIF

          ENDIF

  100	CONTINUE

* No identical line exists
	TM_Find_like_line = unspecified_int4

 	RETURN
	END
