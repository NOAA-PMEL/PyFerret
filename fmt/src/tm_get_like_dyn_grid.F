	SUBROUTINE TM_GET_LIKE_DYN_GRID ( ingrid, its_dyn, outgrid,
     .					  static_lim,  status)

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
* Will compare "ingrid" to other grid definitions in common. Unlike
* tm_get_like_grid this will check both static and dynamic grids.
* If there is a match then "outgrid" returns with the number of the match.
* The variable its_dyn tells if it is dynamic
* If there is no match, then a new dynamic grid is created.
* "outgrid" returns with the number of the new dynamic grid and its_dyn
* is .TRUE..

* created 9/95 *sh*
* 	 11/95 added argument static_lim - kludge because the TMAP library
*	       routines do not know that the upper portion of the grid
*	       COMMON is used as scratch space by Ferret when determining
*	       grids for variables.  This argument prevents accidental
*	       matches against those scratch grids

* Arguments
	LOGICAL		its_dyn
	INTEGER		ingrid, outgrid, static_lim, status
*
* INCLUDE FILES
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif
*
* Local definitions
	INTEGER		grid_match, idim
	INTEGER		TM_FIND_LIKE_GRID, TM_FIND_LIKE_DYN_GRID

* initialize
	its_dyn = .TRUE.

* does it exist already among the existing, static grids?
	outgrid = TM_FIND_LIKE_GRID(ingrid)
	IF (outgrid .NE. unspecified_int4
     ..AND. outgrid .LT. static_lim       ) THEN
* ... yes - it matches an explicit grid
	   its_dyn = .FALSE.
	ELSE

* does it exist already among the existing, dynamic grids?
	   outgrid =  TM_FIND_LIKE_DYN_GRID( ingrid )
	   IF (outgrid .NE. unspecified_int4) THEN
* ... yes - it matches an existing dynamic grid - inc its use count
	      CALL TM_USE_DYN_GRID( outgrid )
	   ELSE

* it shall be a new dynamic grid  "(Gnnn)"
	      CALL TM_ALLO_DYN_GRID( outgrid, status )
	      IF (status .NE. merr_ok) GOTO 9000
	      CALL TM_COPY_GRID_W_LINE_USE( ingrid, outgrid )

* name it "(Gnnn)"
	      WRITE ( grid_name(outgrid), 3000 ) outgrid - max_grids
 3000	      FORMAT('(G',I3.3,')')		! restricted to 3 digits

	   ENDIF
	ENDIF

* successful completion
	status = merr_ok
 9000	RETURN
	END
