	SUBROUTINE TM_deallo_dyn_grid( grid )

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

* Steve Hankin  NOAA/PMEL/TMAP

* 9/13/95 for Ferret version 4.2

* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'tmap_format/xio.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
#endif

* Argument definitions
	INTEGER grid

* Local definitions
	INTEGER next_used_grid, idim

* must be a dynamic grid
	IF ( grid .LE. max_grids ) THEN
	   CALL TM_NOTE('Intern err: TM_DEALLO_DYN_GRID:1 !!!', lunit_errors)
	   RETURN
	ENDIF

* decrement the used count for this grid
	grid_use_cnt(grid) = grid_use_cnt(grid) - 1

* does anyone still have a claim on this grid?
	IF ( grid_use_cnt(grid) .GT. 0 ) THEN
	   RETURN

	ELSEIF ( grid_use_cnt(grid) .EQ. 0 ) THEN

*release claim to any dynamic lines this may use
	   DO 100 idim = 1, 4
	      IF ( grid_line(idim,grid) .GT. max_lines )
     .			CALL TM_DEALLO_DYN_LINE( grid_line(idim,grid))
 100	   CONTINUE

* take this grid from the "used" list and put it on the "free" list
* ... hook onto free list
	   next_used_grid   = grid_flink(grid)
	   grid_flink(grid) = grid_free_ptr  
	   grid_free_ptr    = grid
	   
* ... unhook from used list
	   grid_flink(grid_blink(grid)) = next_used_grid
	   grid_blink(next_used_grid)   = grid_blink(grid)

	ELSE
* internal error -- the use count is negative!!
	   CALL TM_NOTE('Intern err: TM_DEALLO_DYN_GRID:2 !!!', lunit_errors)
	ENDIF

* successful completion
	RETURN

	END
