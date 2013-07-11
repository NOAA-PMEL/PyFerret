
	SUBROUTINE DELETE_ONE_LINE (axis, status)
	
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

* Delete a single axis (from xeq_cancel)

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
        include 'xtm_grid.cmn_text'

	INTEGER axis, slen, status
	CHARACTER*10 TM_FMT, buff
	REAL	raxis

	INTEGER TM_GET_GRID_OF_LINE, TM_GET_LINENUM, 
     .		grid, num_protected_axes

	raxis = axis
	buff = TM_FMT(raxis, 6, 10, slen)
	num_protected_axes = TM_GET_LINENUM( 'EZ' )  ! top protected axis

	IF ( axis .EQ. unspecified_int4 ) GOTO 6210
	IF ( axis .LE. num_protected_axes ) GOTO 6212
	line_keep_flag(axis) = .FALSE.  ! see tm_garb_col_grids.F
	IF (line_use_cnt(axis) .LE. 0) THEN
	   IF ( axis .LE. max_lines ) THEN
	      IF ( axis .LE. max_lines ) THEN
	         IF (.NOT. line_regular(axis))
     .                CALL PACK_LINE_STORAGE(axis)
	         line_name(axis) = char_init16
	      ELSE
	         CALL TM_DEALLO_DYN_LINE(axis)
	      ENDIF
	   ENDIF
	ELSE
*     ... cannot delete - in use.  Locate the grid that is using it.
	   grid = TM_GET_GRID_OF_LINE( axis )
	   CALL WARN( 'Not deleted: '//line_name(axis))
	   IF ( grid .NE. unspecified_int4 ) THEN
	      CALL WARN( 
     .		'Axis is in use by grid '//grid_name(grid))
	   ELSE
	      CALL ERRMSG(ferr_internal, status,
     .				'axis use count err', *5000)
	   ENDIF
	ENDIF

 5000	CONTINUE
	RETURN

 6210	CALL ERRMSG( ferr_invalid_command, status, 'unknown axis: '//
     .		     buff(:slen), *5000 )
 6212	CALL ERRMSG( ferr_invalid_command, status, 'protected axis: '//
     .		     buff(:slen), *5000 )

	END
