
	SUBROUTINE DEALLO_ALL_AXES

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
* Deallocates all axis (line) slots defined by user 
* Does not deallocate the first group of "protected" axes.
* ACM 6/11/2002
*
* V5.41 7/30/02 ACM Deallocate irregular dynamic axes defined by files too,
*               by calling remove_axis.
* V5.42 10/31/02ACM Check to see if axes are in use, and if so dont remove!


* Parameters and commons

	include	'tmap_dims.parm'
#	include	"tmap_dset.parm"
	include	'tmap_errors.parm'
	include 'xtm_grid.cmn_text'
        include 'ferret.parm'
	include 'errmsg.parm'
	external xgt_grid_data

* Local declarations
	INTEGER TM_GET_LINENUM, TM_GET_GRID_OF_LINE,
     .          islot, grid, status, num_protected_axes

* Remove axis names

	num_protected_axes = TM_GET_LINENUM( 'EZ' )  ! top protected axis

        IF (num_protected_axes .EQ. unspecified_int4 .OR.
     .      num_protected_axes .LE. 0) num_protected_axes = 1

        DO 100 islot = num_protected_axes+1, line_ceiling
           if (islot .gt. 1000 .and. islot .lt. 1006) then
              status = 0
           endif
           IF (line_name(islot) .NE. char_init16) THEN

              line_keep_flag(islot) = 0
              IF (line_use_cnt(islot) .LE. 0) THEN
	         IF ( islot .LE. max_lines ) THEN
                    IF (.NOT. line_regular(islot)) CALL REMOVE_AXIS(islot)
		    line_name(islot) = char_init16
	         ELSE
	            CALL TM_DEALLO_DYN_LINE(islot)
	         ENDIF
              ELSE

*     ... cannot delete - in use.  Locate the grid that is using it.

	         grid = TM_GET_GRID_OF_LINE( islot )
	         CALL WARN( 'Not deleted: '//line_name(islot))
	         IF ( grid .NE. unspecified_int4 ) THEN
	            CALL WARN( 
     .		      'Axis is in use by grid '//grid_name(grid))
	         ELSE
	            CALL ERRMSG(ferr_internal, status,
     .				'axis use count err', *5000)
	         ENDIF
	      ENDIF

           ENDIF

  100	CONTINUE

	RETURN

* error exits
 5000	RETURN

	END
