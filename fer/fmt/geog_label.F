	LOGICAL FUNCTION GEOG_LABEL( idim, grid )

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
* determine whether a particular axis of a grid represents geographical
* locations (lat.,long., etc.) requiring special label formatting

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/25/88
* V230: 4/92 *sh*  only return .TRUE. for time axis if t0_date is valid
*       7/92 - use TM_DATE_OK for this
*            - vertical axis requires meters units AND UD orientation
* V320: 12/19/94 - recognize millibars and decibars as special axes
* v491: 2/23/98 *kob* - add check for undefined axis which can cause
*                       a crash if REPEAT command doesn't have fully 
*                       mature context to work with.

#ifdef unix
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
        include 'tmap_errors.parm'
	external xgt_grid_data
        include 'xunits.cmn_text'
        external xunits_data
	include 'ferret.parm'
	include 'xtext_info.cmn'
	include 'xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER		idim, grid

* internal variable declarations:
        LOGICAL         TM_DATE_OK
	INTEGER		i, line
	CHARACTER*2	line_code

* check for program bug	! temp
	IF ( grid .EQ. unspecified_int4 ) STOP 'no_grd_orient'

* see if labeling modes are disabled, first
	IF ( .NOT.ax_fmt(idim) ) THEN
	   GEOG_LABEL = .FALSE.
           RETURN
	ENDIF

* initialize
	line = grid_line( idim, grid )
* 2/98 *kob* add check for undefined axis 
	IF ( line .EQ. mnormal .OR. line .EQ. munknown 
     .         .OR. line .EQ. unspecified_int4) THEN
	   GEOG_LABEL = .FALSE.
	   RETURN
	ELSE
	   line_code = line_direction( line )
	ENDIF

* the use of line_direction is kludgy
* (it combines orientation, reversal, and formatting in an inadequate way)
        IF ( idim .LE. 2 ) THEN
           GEOG_LABEL = line_code .EQ. axis_orients(idim)
        ELSEIF ( idim .EQ. z_dim ) THEN
* for Z also insist on units of m because DU/UD is used to indicate reversal
           GEOG_LABEL =((line_code .EQ. 'UD'
     .            .AND. line_unit_code(line) .EQ. pun_meters)
     .    .OR.		(line_code .EQ. 'UD'
     .		   .AND. line_unit_code(line) .EQ. pun_millibars)
     .    .OR.		(line_code .EQ. 'UD'
     .		   .AND. line_unit_code(line) .EQ. pun_decibars) )
        ELSE
*  4/92 addition:  check for valid time axis T0
           GEOG_LABEL = line_code        .EQ. 'TI'
     .            .AND. TM_DATE_OK( line_t0(line) )
        ENDIF

	RETURN
	END
