	LOGICAL FUNCTION GEOG_VAR( idim, cx )

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
* determine whether a particular axis of a VARIABLE represents geographical
* locations (lat.,long., etc.) requiring special label formatting

* V450: 3/97 - based on GEOG_LABEL

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
	include 'xcontext.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		idim, cx

* internal variable declarations:
        LOGICAL         TM_DATE_OK
	INTEGER		TM_UNIT_ID, units
	CHARACTER*80	VAR_UNITS

* initialize
	GEOG_VAR = .FALSE.	! default

* see if labeling modes are disabled, first
	IF ( .NOT.ax_fmt(idim) )  RETURN

* get the "units" that goes with the variable (context)
	units = TM_UNIT_ID( VAR_UNITS(cx) )

* do these units imply a formatt-ible axis
	IF ( idim.EQ.t_dim .AND.
     .        (units.LT.pun_1st_time.OR.units.GT.pun_last_time)) THEN
	   GEOG_VAR = .FALSE.
!	   GEOG_VAR = .TRUE.		! where to get T0 ???
	ELSEIF ( (idim.EQ.x_dim.OR.idim.EQ.y_dim) .AND.
     .		units .EQ. pun_degrees ) THEN
	   GEOG_VAR = .TRUE.
	ELSEIF ( (idim.EQ.z_dim) .AND.
     .          ( units .EQ. pun_millibars
     .          .OR. units .EQ. pun_decibars
     .          .OR. units .EQ. pun_meters )  ) THEN
	   GEOG_VAR = .TRUE.
	ENDIF

	RETURN
	END
