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
* Define an axis for the aggregation direction (Ensemble or Forecast). If more
* than one aggregation axis is defined, TM_NEW_LINE_NAME makes a unique name.

* original: *acm*
* V695+ *sh* 5/15 - set up calendar axis attributes from 1st member dset
* V698 *sh* 1/16 - use TM_ALLO_TMP_LINE to create axis, so its use counts get 
*			set and it gets deallocated automatically
*      *sh* 2/16 - if a pre-existing identical axis exists, re-use it
*                  else we cannot aggregate the aggs (for EF and FE aggs)
! considered but rejected this change:
!*      *sh* 3/16 - shift E coords down so that EBOXLO[m=1] is at zero

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
	INTEGER TM_FIND_LIKE_DYN_LINE, pre_exist
	CHARACTER*64 newname

* create the ensemble or forecast axis
	CALL TM_ALLO_TMP_LINE (iline, status)
	IF ( status .NE. ferr_ok ) GOTO 5000

* save the line in memory - but don't catalog its name
	IF (line_dir .EQ. e_dim) THEN
	   CALL TM_NEW_LINE_NAME ( 'ENSEMBLE', newname )
	   line_direction( iline ) = 'EE'
	   line_units( iline ) = plag_ax_units_e
	ENDIF
	IF (line_dir .EQ. f_dim) THEN
	   CALL TM_NEW_LINE_NAME ( 'RUN', newname )
	   line_direction( iline ) = 'FI'
	   line_units( iline ) = plag_ax_units_f
	ENDIF

	line_name     ( iline ) = newname
	line_name_orig( iline ) = newname
	line_start    ( iline ) = 1.0D0
!	line_start    ( iline ) = 0.5D0     ! something to consider ...3/16
	line_delta    ( iline ) = 1.0D0
	line_dim      ( iline ) = len
	line_regular  ( iline ) = .TRUE.
	line_modulo   ( iline ) = .FALSE.
	line_unit_code( iline ) = 0			! unknown units
	line_shift_origin( iline ) = .FALSE.

!	line_direction( iline ) = 'NA'
!	IF (line_dir .EQ. x_dim) THEN
!	   line_direction( iline ) = 'WE'
!	   line_name     ( iline ) = 'LONGUITUDE'
!	ELSEIF (line_dir .EQ. y_dim) THEN
!	   line_direction( iline ) = 'SN'
!	   line_name     ( iline ) = 'LATITUDE'
!	ELSEIF (line_dir .EQ. z_dim) THEN
!	   line_direction( iline ) = 'UD'
!	   line_name     ( iline ) = 'DEPTH'
!	ELSEIF (line_dir .EQ. t_dim) THEN
!	   line_direction( iline ) = 'TI'
!	   line_name     ( iline ) = 'TIME'
!	ELSEIF (line_dir .EQ. e_dim) THEN
!	   line_direction( iline ) = 'EE'
!	   line_name     ( iline ) = 'ENSEMBLE'
!	ELSEIF (line_dir .EQ. f_dim) THEN
!	   line_direction( iline ) = 'FI'
!	   line_name     ( iline ) = 'FORECAST'
!	ENDIF
	line_name_orig( iline ) = line_name ( iline )

! this code not needed.  AGG_CLEAN_GRIDS takes care of removing dups
!* now check to see if an identical axis already exists.  If so, then
!* use the pre-existing one, instead
!	pre_exist = TM_FIND_LIKE_DYN_LINE( iline )
!	IF (pre_exist .NE. unspecified_int4) THEN
!	   line_name (iline) = char_init16   ! deallo'd in tm_garb_collect
!!	   CALL TM_DEALLO_DYN_LINE(iline)
!	   iline = pre_exist
!	ENDIF

 5000   RETURN	
 5300   status = ferr_TMAP_error
	RETURN

	END
