	SUBROUTINE COMPLETE_MISSING_LIMITS( cx )


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


* Replace unspecified limits in the given context with full-span limits
* (not clear how cx_given sould be set ...)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

*  V500 *sh* 3/99
*       *sh* 4/99 - bug fix: ABSTRACT axes completed on some user vars

#ifdef unix
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xfr_grid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER	cx

* internal variable declarations:
	LOGICAL needs_limits, its_unspec
	INTEGER idim, cat, var, grid, mabstract, line, status

* initialize
	var  = cx_variable( cx )
	cat  = cx_category( cx )
	grid = cx_grid    ( cx )

* get the axis number for the abstract axis
        mabstract = grid_line( x_dim, mgrid_abstract )

	DO 100 idim = 1, 4

	   IF ( cx_by_ss(idim, cx) ) THEN
	      its_unspec = cx_lo_ss(cx,idim) .EQ. unspecified_int4
	   ELSE
	      its_unspec = cx_lo_ww(idim,cx) .EQ. unspecified_val8
	   ENDIF

	   IF ( its_unspec ) THEN

	      line = grid_line( idim, grid )

	      needs_limits = line .NE. mnormal
     .		       .AND. line .NE. mabstract
	      IF ( needs_limits .AND. cat.EQ.cat_user_var ) THEN
	         needs_limits = uvar_given(idim,var) .GE. uvlim_needed
	      ENDIF

	      IF ( needs_limits ) THEN

	         cx_by_ss(idim, cx) = .TRUE.
	         IF ( cat .EQ. cat_file_var ) THEN
	            CALL VAR_SS_LIMS( idim, cx,
     .				      cx_lo_ss(cx,idim),
     .				      cx_hi_ss(cx,idim)  )
	         ELSE
	            cx_lo_ss(cx,idim) = 1
	            cx_hi_ss(cx,idim) = line_dim( line )
	         ENDIF

*  set cx_given to match behavior of "SET REGION limits"
	         cx_given(idim, cx) = .FALSE.

* flesh out the world coordinates, too
	         CALL  FLESH_OUT_AXIS( idim, cx, status )
* ... status ignored ...

	      ENDIF

	   ENDIF

 100	CONTINUE

	RETURN

	END
