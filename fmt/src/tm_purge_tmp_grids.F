	SUBROUTINE TM_PURGE_TMP_GRIDS( status )

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

* This routine is normally called only for error conditions that occur
* in the midst of building grids and axes for netCDF files.
* The goal is to throw away all of the in-progress work on grids and
* axes, including allocations (use count increments) of axes that came
* from other data sets.

* Steve Hankin  NOAA/PMEL/TMAP

* V510 *sh* 3/00

* Include files
	include 'tmap_dims.parm'
#include 'gt_lib.parm'
	include 'xtm_grid.cmn_text'
	include 'xio.cmn_text'


* Argument definitions
	INTEGER status

* Local variable declarations
	INTEGER grid, line, idim


* over temporary grids
 100	CONTINUE
	   grid = 0	! to get start of list
	   CALL TM_NEXT_TMP_GRID(grid, *300)

* deallocate axes used by grid (NOTE: only dynamic axes are used by netCDF!)
	   DO 200 idim = 1, 4
	      IF ( grid_line(idim, grid) .GT. max_lines ) THEN
	         CALL TM_DEALLO_DYN_LINE( grid_line(idim, grid) )
	      ELSEIF (  grid_line(idim, grid) .NE. mpsnorm ) THEN
	         CALL TM_NOTE('tm intern err: netCDF grid w/ static axis'
     .			 ,lunit_errors)  ! should never happen
	      ENDIF
 200	   CONTINUE

	   CALL TM_DEALLO_DYN_GRID( grid )
	GOTO 100

* remove any stray lines that were never incorporated into grids
 300	CONTINUE
	   line = 0		! to get start of list
 	   CALL TM_NEXT_TMP_LINE(line, *1000)

* deallocate axes used by grid (NOTE: only dynamic axes are used by netCDF!)
	   DO 400 idim = 1, 4
	      IF ( line .GT. max_lines ) THEN
	         IF (line_use_cnt(line) .GT. 1) CALL TM_NOTE(
     .			'tm intern err: line over alloc.',lunit_errors)
	         CALL TM_DEALLO_DYN_LINE( line )
	      ELSEIF (  line .NE. mpsnorm ) THEN
	         CALL TM_NOTE('intern err: netCDF static axis'
     .			 ,lunit_errors)  ! should never happen
	      ENDIF
 400	   CONTINUE

	GOTO 300

* successful completion
 1000	RETURN

 	END
