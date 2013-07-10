	SUBROUTINE CREATE_AGG_AXIS (len, line_dir, iline, status)

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
* Define an axis for the aggregation direction (Ensemble, for now). If more
* than one ensemble axis is defined, make the names unique.

      IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	
* calling argument declarations:
	INTEGER	len, line_dir, iline, status

* local variable declarations:
	INTEGER ALLO_MANAGED_AXIS
	CHARACTER*64 ename

* create the ensemble axis

	status = ALLO_MANAGED_AXIS (iline )
	IF ( status .NE. ferr_ok ) GOTO 5000

* save the line in memory - but don't catalog its name
	CALL TM_NEW_LINE_NAME ( 'ENSEMBLE', ename )

	line_name     ( iline ) = ename
	line_name_orig( iline ) = ename
	line_start    ( iline ) = 1.0D0
	line_delta    ( iline ) = 1.0D0
	line_dim      ( iline ) = len
	line_regular  ( iline ) = .TRUE.
	line_modulo   ( iline ) = .FALSE.
	line_units    ( iline ) = ' '
	line_unit_code( iline ) = 0			! unknown units
	line_shift_origin( iline ) = .FALSE.

	line_direction( iline ) = 'NA'
	IF (line_dir .EQ. x_dim) THEN
	   line_direction( iline ) = 'WE'
	   line_name     ( iline ) = 'LONGUITUDE'
	ELSEIF (line_dir .EQ. y_dim) THEN
	   line_direction( iline ) = 'SN'
	   line_name     ( iline ) = 'LATITUDE'
	ELSEIF (line_dir .EQ. z_dim) THEN
	   line_direction( iline ) = 'UD'
	   line_name     ( iline ) = 'DEPTH'
	ELSEIF (line_dir .EQ. t_dim) THEN
	   line_direction( iline ) = 'TI'
	   line_name     ( iline ) = 'TIME'
	ELSEIF (line_dir .EQ. e_dim) THEN
	   line_direction( iline ) = 'EE'
	   line_name     ( iline ) = 'ENSEMBLE'
	ELSEIF (line_dir .EQ. f_dim) THEN
	   line_direction( iline ) = 'FI'
	   line_name     ( iline ) = 'FORECAST'
	ENDIF
	line_name_orig( iline ) = line_name ( iline )
	
 5000   RETURN	
 5300   status = ferr_TMAP_error
	RETURN

	END
