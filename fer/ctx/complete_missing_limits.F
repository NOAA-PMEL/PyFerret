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
*	*sh* 4/99 - add mode_stupid check
*	*sh* 5/99 - complete limits on implicit axes
*  V510 *sh* 4/00 - netCDF variables may now have dynamic axes, so chg the
*		    logic that checked for these
*  V520 *sh* 6/00 - a cmnd with unspecified lims a conflicting X axes like
*			LIST Xshort-Xlong
*		    needs to generate an error. It does not because this
*		    routine requests I=1:short from both components.
*		    Change cx_by_ss to FALSE in here -- meaning that the
*		    misleading behavior can occur only when each component
*		    has the same number of points in the same world coord span

#ifdef unix
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'implicit.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
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

* possible skip this routine
	IF (mode_stupid) GOTO 1000

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

 11	    needs_limits = line .NE. mnormal
     .		       .AND. line .NE. mabstract

* If the limits are given in the definition of the variable (e.g. A=B[J=3:5])
* then it is not possible to complete the limits from the axis length, alone.
* (5/99) However, do complete implicit axis ( >=max_lines ) lims, since
* implicit axes are always full length. This will improve cache hits on
* LOAD a, where a=netCDFvar[i=1:10:2] which uses netCDF strides
* WARNING: For b=a[i=10:100:10];c=b[i=3:5] we have an implicit axis where the
* axis length is NOT applicable. HOPEFULL completing these limits will not
* cause a problem.
* Note: 4/00 Since netCDF axes are now dynamic we can no longer use max_lines
*	as a test of anything significant. I replaced the restriction 
*	    " .	    .AND.  line .LT. max_lines ) THEN"
*	with a test on line_parent. ALERT!

	      IF ( needs_limits .AND. cat.EQ.cat_user_var
     .	    .AND.  line_class(line) .NE. pline_class_stride ) THEN
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

* matches to this region must be based on world coordinates (see 6/00 comments)
	         cx_by_ss(idim, cx) = .FALSE.


	      ENDIF

	   ENDIF

 100	CONTINUE

 1000	RETURN

	END
