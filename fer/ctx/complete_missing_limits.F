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
*		    MODE STUPID:weak_cache in case future EF problems arise
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
*
*  V541 *kob* 10/02 - BUG FIX - need to check the axis number for the 
*                               abstract grid for EACH dimension - not
*                               just x dimension, in order to correctly
*                               have limits set
* V550 *sh* 11/02 - added remark about weak_cache to 4/99 comments
*			(remark copied from ChangeLog)
* ??? v570 *acm* 7/04  do we want to use mabstract = grid_line( idim, saved_abstract_grid )
*                      instead of getting the line from mgrid_abstract ???
* v602 *acm* 12/06 use TM_GET_GRIDNUM ('ABSTRACT') to get the grid number of the grid
*                  of abstract axes; mgrid_abstract and saved_abstract_grid refer to the
*                  grid of a SET GRID gridname command.
* v6.11 *acm* 4/08 Fix bug 1558: revert previous changes.
*                  When we WANT the grid of a SET GRID gridname command, need to use
*                  mgrid_abstract and saved_abstract_grid. I no longer know the reason 
*                  for the previous change; it does not seem to fix any listed bug and
*                  reverting does not break anything...
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V690 1/14 *sh* removed special behavior for mode_stupid
* V698 3/16 *sh* added logic for cat_ef_time2d (ACTS_LIKE_FVAR)
* V720 4/17 *sh* added logic for DSG ragged contiguous grids

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'implicit.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xfr_grid.cmn'

* calling argument declarations:
	INTEGER	cx

* internal variable declarations:
	LOGICAL ACTS_LIKE_FVAR, TM_ITSA_DSG_RAGGED,
     .		needs_limits, its_unspec, its_dsg, its_cmpnd
	INTEGER TM_GET_GRIDNUM,
     .          idim, cat, var, grid, mabstract, line, status, grr,
     .		dsg_obsdimlen, dsg_orientation, dsg_line

* possible skip this routine
! commented this out 1/2014 cuz it causes a crash in aux var processing
! and it isn't clear why Ferret should function correctly when this line is here
!	IF (mode_stupid) GOTO 1000

* initialize
	var  = cx_variable( cx )
	cat  = cx_category( cx )
	grid = cx_grid    ( cx )

        grr = TM_GET_GRIDNUM ('ABSTRACT')

* *** trap DSG grids for special treatment
	its_dsg = TM_ITSA_DSG_RAGGED(grid)
	IF (its_dsg) CALL TM_DSG_FACTS(grid,
     .		     dsg_orientation, dsg_obsdimlen, 
     .		     dsg_line, its_dsg, its_cmpnd)
     
	IF (its_dsg) THEN
* ... always set up to read the full 1D obs dim of the variable
*     Subsetting is done with DEFINE DATA/SUBSET
	   
	   DO idim = 1, nferdims
	      cx_lo_ss(cx,idim) = unspecified_int4
	      cx_hi_ss(cx,idim) = unspecified_int4
	   ENDDO
	   IF (its_cmpnd) dsg_orientation = z_dim
	   cx_lo_ss(cx,dsg_orientation) = 1
	   cx_hi_ss(cx,dsg_orientation) = dsg_obsdimlen
* ... leave the world coordinates as-is
	   RETURN
	ENDIF
* ***

	DO 100 idim = 1, nferdims

* get the axis number for the abstract axis

           grr = saved_abstract_grid
           IF (grr .EQ. unspecified_int4) grr = mgrid_abstract
	   mabstract = grid_line( idim, mgrid_abstract )

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
                 IF ( ACTS_LIKE_FVAR(cat) .OR. 
     .                (cat .EQ. cat_pystat_var) ) THEN
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
