	SUBROUTINE CHECK_NEG_TAXIS( dset )

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
* check to make sure that the time axis of the data set doesn't have any
* negative time step values

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/11/90
* Unix/RISC port:  07/91 - allow normal time axis
*                  10/91 - briefer warning
* V230 - simplified (less intimidating) message
*      7/20/92 - don't issue warning unless T0 is valid
*              - (cleaned up unused variables)
* V530  1/01 *acm* cal_id for alternative calendar axis types
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
* calling argument declarations:
	INTEGER	dset

* internal variable declarations:
        LOGICAL TM_DATE_OK
	INTEGER	i, grid, line
        INTEGER TM_GET_CALENDAR_ID, cal_id
	DOUBLE PRECISION TM_WORLD

	include 'ferret.parm'
	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* base it on first variable in data set
	DO i = 1, maxvars
	   IF ( ds_var_setnum(i) .EQ. dset ) GOTO 110
 	ENDDO
	RETURN	! no vars in set

* check lowest time step value on axis
 110	grid = ds_grid_number(i)
        line = grid_line(t_dim, grid )

* allow time-independent grids (7/91)
        IF ( line .EQ. mnormal ) RETURN

* no warning for non-date axes - this way time can be a regular ol' axis

        cal_id = TM_GET_CALENDAR_ID (line_cal_name(line) )
        IF ( .NOT.TM_DATE_OK( line_t0(line), cal_id) ) RETURN

	IF ( TM_WORLD(1,grid,t_dim,box_middle) .LT. 0.0D0 ) THEN
	   CALL WARN('Negative values in time axis'//
     .               ' may not be formatted properly: '//
     .		     ds_name(dset) )

	ELSEIF ( TM_WORLD(1,grid,t_dim,box_lo_lim) .LT. 0.0D0 ) THEN
	   CALL WARN('Negative values at start of time axis'//
     .               ' may be improperly formatted: '//
     .		     ds_name(dset) )
	ENDIF

	RETURN
	END
