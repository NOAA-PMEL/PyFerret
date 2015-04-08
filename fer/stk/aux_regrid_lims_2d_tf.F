	SUBROUTINE AUX_REGRID_LIMS_2D_TF( dst_cx, src_cx, axis_list, naxis,
     .			                 aux1, mr1, cx1, status )

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
* This routine handles the special case of forecast model aggregations
* It determines the range of souce data on the F and T axes in the 2D time
* form that is needed to create calendar-time x lag-forecast form

* In generic terms, the descript of the routine is this:
* Given a context describing data to be regridded to a destination grid
* create the description of the needed source data in the source context
* Use the auxiliary variables to determine the index range of data needed

* The 2D time field is the auxiliary regrid control variable. It contains
* times that are partially duplicated from one forecat index to the next
* The orthoganalization process removes the duplication by shifting the start
* time index of each successive forcast, so that all forecasts lie on the
* same-single time axis

* When called the source context should be a copy of the destination context 
* This routine is almost a copy of AUX_REGRID_LIMS_1D_T.  The difference is
* that the user cannot limit the N range of source data needed through the /N 
* or /F qualifier.  Instead this code determines the N range needed to fullfill
* the requested time range

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

*v695+ 3/15 *sh* almost a copy of AUX_REGRID_LIMS_1D_T

	IMPLICIT NONE

	include 'tmap_dims.parm'
	include 'implicit.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xtm_grid.cmn_text'   ! for line_delta and line_regular

* calling argument declarations:
	INTEGER	src_cx, dst_cx, naxis, axis_list(naxis), status,
     .		mr1, cx1, mr2, cx2, mr3, cx3 
        REAL*8  aux1(m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )

* internal variable declarations:
	INTEGER	KNOWN_GRID, CGRID_AXIS,
     .		idim, src_grid, i, j, k, l, m, n, trans,
     .		ndim, dim(nferdims), looking_for_first,
     .		ldstmin, ldstmax, llmin, llmax, llo, lhi, ldel,
     .		nlo, nhi, nmin, nmax, axis
	REAL*8  TM_WORLD, axmax, axmin, bad_aux, t, t1, steps,
     .		tdstrt, tdend, tstepmin, tstepmax
!, f_delta, t_delta

* initialize
	bad_aux = mr_bad_data( mr1 )
!	idim     = axis_list(1)
	llo      = cx_lo_s4(cx1)   ! 2d time array limit - should always be 1
	lhi      = cx_hi_s4(cx1)   ! should always be N
	nlo      = m1lof	   ! should always be 1
	nhi      = m1hif	   ! should always be dimension size of F
	i        = cx_lo_s1(cx1)   ! fixed index since aux field is 2d
	j        = cx_lo_s2(cx1)
	k        = cx_lo_s3(cx1)
	m        = cx_lo_s5(cx1)
	tstepmin = aux1(i,j,k,llo,m,nlo) ! low-left corner of 2d time array
	tstepmax = aux1(i,j,k,lhi,m,nhi) ! upper right
	tdstrt   = TM_WORLD(cx_lo_s4(dst_cx), cx_grid(dst_cx),
     .			  t_dim, box_middle) ! 'd' for destination start time
	tdend    = TM_WORLD(cx_hi_s4(dst_cx), cx_grid(dst_cx),
     .			  t_dim, box_middle)
	llmin  = llo	! initial default
	llmax  = lhi
	nmin   = nlo
	nmax   = nhi 		

* insert source grid (the defining grid for this variable)
* XXXX TEMPORARY -- ?move this block of code to AUX_VAR_REGRID when 2D curv is added?
	src_grid = KNOWN_GRID( cx_data_set(src_cx),
     .			       cx_category(src_cx),
     .			       cx_variable(src_cx)  )
	cx_grid( src_cx ) = src_grid
	cx_unstand_grid( src_cx ) = .FALSE.
	is_phase( isp )   = prgrd_from_source
	cx_naux( src_cx ) = 0
        DO 100 idim = 1, nferdims
           cx_aux_var  (idim, src_cx) = unspecified_int4
           cx_aux_cat  (idim, src_cx) = unspecified_int4
           cx_aux_stat (idim, src_cx) = paux_stat_na
 100    CONTINUE

* commented out these checks for regular time steps, because of monthly axes
!* sanity check the target T axis
!	axis = CGRID_AXIS(t_dim,dst_cx)
!	IF (.NOT. line_regular(axis)) GOTO 5050
!	IF (lhi .GT. llo) THEN
!	   IF (line_delta(axis)
!     .	   .NE.(aux1(i,j,k,llo+1,m,nlo)-aux1(i,j,k,llo,m,nlo))) GOTO 5055
!	ENDIF
!
* sanity check the target F axis
	axis = CGRID_AXIS(f_dim,dst_cx)
	IF (.NOT. line_regular(axis)) GOTO 5060
!	IF (nhi .GT. nlo) THEN
!	   f_delta = line_delta(axis)
!	   IF ( f_delta
!     .	   .NE.(aux1(i,j,k,llo,m,nlo+1)-aux1(i,j,k,llo,m,nlo))) GOTO 5065
!	ENDIF


* ideally we should check that the units of the 2D time field are the same as the units of the
* 1D time axis, but since typically Ferret code is creating the 1d time, we'll allow laziness

* this routine only for t_dim -- index-based remapping of time locations
	trans = cx_regrid_trans(t_dim,src_cx)
	IF (trans .NE. pauxrgrd_fmrc) 
     .	  CALL ERRMSG( ferr_internal, status, 'unkwn rgrd trans', *5000)

* make sure that the auxiliary variable is monotonic
* (not regularly spaced cuz we havvvvvvvvvve to support monthly data)
*	The 2D time array can never theoretically be non-monotonic
*       FT aggregations are new and unreliable
	DO n = nlo, nhi
	   looking_for_first = 0
           DO l = llo, lhi
	      t = aux1(i,j,k,l,m,n)
	      IF (t .EQ. bad_aux) THEN
	         GOTO 5150
	      ELSEIF (looking_for_first .EQ. 0) THEN
	         t1 = t
	         looking_for_first = 1
	      ELSE
	         IF (looking_for_first .EQ. 1) THEN
!	            t_delta = t-t1
!	            IF (t_delta .LE.0.0) GOTO 5100
	            IF (t .LE. t1) GOTO 5100
	         ENDIF
!	         IF ((t-t1) .NE. t_delta) GOTO 5100  ! need not be regular
	         t1 = t
	      ENDIF
	   ENDDO
	ENDDO

!* the delta in F must be an integer multiple of the delta in T
!	steps = f_delta / t_delta
!	IF (steps .NE. INT(steps)) GOTO 5200

* the relationship between the time steps in the 2D aux time field (cx1)
* and the same times in the destination field looks something
* like this

*          **  2d aux times    **  ==>   ** cal-T x lag-F field **

*   12  1011(0)   1015(1)   1019(2)        1019(2) 1019(3) 1019(4)     20
*   11  1010(0)   1014(1)   1018(2)        1018(2) 1018(3) 1018(4)     19
*   10  1009(0)   1013(1)   1017(2)        1017(2) 1017(3) 1017(4)     18
*    9  1008(0)   1012(1)   1016(2)        1016(2) 1016(3) 1016(4)     17
*    8  1007(0)   1011(1)   1015(2)        1015(1) 1015(2) 1015(3)     16
*    7  1006(0)  >1010(1)   1014(2)        1014(1) 1014(2) 1014(3)     15
*    6  1005(0)   1009(1)   1013(2)        1013(1) 1013(2) 1013(3)     14
*    5  1004(0)   1008(1)   1012(2)        1012(1) 1012(2) 1012(3)     13
*    4  1003(0)   1007(1)   1011(2)        1011(0) 1011(1) 1011(2)     12
*    3  1002(0)   1006(1)   1010(2)        1010(0) 1010(1) 1010(2)     11
*    2  1001(0)  >1005(1)   1009(2)        1009(0) 1009(1) 1009(2)     10
*    1  1000(0)   1004(1)   1008(2)        1008(0) 1008(1) 1008(2)      9
*    ^                                                                  ^
*    |                                                                  |
* l in src_cx                                                     l in dst_cx 

* say that the requested time range is T=1005:1010 (marked above)
* The lowest L index of source data needed is the index at which the START time
*    is first encountered as we loop down the forecasts searching lo to hi
* The highest L index of source data needed is the index at which the END  time
*    is first encountered as we loop up   the forecasts searching hi to lo

*************************************************************************
*!!! >>> EFFICIENCY NOTE  <<< !!!
* In some important situations this is a very inefficient way to appproach
* the FT re-organization that is performed here. The most important example
* is when the user requests the "best time series" -- i.e. N=1 only, which
* gives a time series of the initial time steps of the model following each
* initialization.  In that case the efficient approach is to individually
* request the sequence of time series fragments and concatenate them.
* (Visualize how to obtain first column of the right hand block above)
* Instead this code requests the full 2D block of data.  Inefficient!
*************************************************************************

* bail out if the requested min exceeds the available range or similar max
	IF (tdstrt .GT. tstepmax
     .  .OR. tdend .LT. tstepmin ) THEN
* ... no regridding is possible - result will be a field of missing data
* ... provide a nominal data region that is not trouble to get
	   llmax = llmin  ! request a single point region only of source data
	   nmax  = nmin
	   GOTO 500
	ENDIF

* determine the minimal forecast index range (i.e. with relevant timesteps)
	DO n = nlo, nhi
	   t  = aux1(i,j,k,llo,m,n)
	   t1 = aux1(i,j,k,lhi,m,n)
	   IF (t1 .GE. tdstrt
     .   .AND. t  .LE. tdend  ) THEN
* ... found first forecast with relevant time steps
	      nmin = n
	      EXIT
	   ENDIF
	ENDDO
	DO n = nhi, nlo, -1
	   t  = aux1(i,j,k,llo,m,n)
	   t1 = aux1(i,j,k,lhi,m,n)
	   IF (t1 .GE. tdstrt
     .   .AND. t  .LE. tdend  ) THEN
* ... found last forecast with relevant time steps
	      nmax = n
	      EXIT
	   ENDIF
	ENDDO
	IF (nmin .GT. nmax) nmax = nmin

* always request the full source time range (entire forecasts)
	llmin = cx_lo_s4(cx1)
	llmax = cx_hi_s4(cx1)

* apply the T limits that have been determined
 500	idim = t_dim
	IF (llmin .GT. llmax) llmax = llmin  ! may not be needed?
	cx_lo_ss ( src_cx, idim )  = llmin
        cx_hi_ss ( src_cx, idim )  = llmax
        cx_by_ss ( idim, src_cx )  = .TRUE.

        cx_lo_ww ( idim, src_cx )  = unspecified_val8
        cx_hi_ww ( idim, src_cx )  = unspecified_val8
        cx_delta ( idim, src_cx )  = unspecified_val8
        cx_trans ( idim, src_cx )  = trans_no_transform
        cx_trans_arg(idim,src_cx)  = bad_val4
        cx_given( idim, src_cx )  = .FALSE.
        cx_regrid_trans(idim,src_cx) = unspecified_int4 
        cx_formatted( idim )   = .FALSE.

	CALL FLESH_OUT_AXIS( idim, src_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* apply the F limits that have been determined
	idim = f_dim
	cx_lo_ss ( src_cx, idim )  = nmin
        cx_hi_ss ( src_cx, idim )  = nmax
        cx_by_ss ( idim, src_cx )  = .TRUE.

        cx_lo_ww ( idim, src_cx )  = unspecified_val8
        cx_hi_ww ( idim, src_cx )  = unspecified_val8
        cx_delta ( idim, src_cx )  = unspecified_val8
        cx_trans ( idim, src_cx )  = trans_no_transform
        cx_trans_arg(idim,src_cx)  = bad_val4
        cx_given( idim, src_cx )  = .FALSE.
        cx_regrid_trans(idim,src_cx) = unspecified_int4 
        cx_formatted( idim )   = .FALSE.

	CALL FLESH_OUT_AXIS( idim, src_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* success
        status = ferr_ok
	RETURN

* error exit
 5000	RETURN
! 5050   CALL ERRMSG( ferr_invalid_command, status,
!     .		'target 1D time axis is not regular', *5000)
! 5055	CALL ERRMSG( ferr_invalid_command, status,
!     .		'delta T of target 1D time axis doesnt match the data', *5000)
 5060   CALL ERRMSG( ferr_invalid_command, status,
     .		'target forecast lag time axis is not regular', *5000)
! 5065	CALL ERRMSG( ferr_invalid_command, status,
!     .		'delta T of forecast lag axis doesnt match the data', *5000)
 5100   CALL ERRMSG( ferr_invalid_command, status,
     .		'2D T-F times are not strictly increasing on T and F', *5000)
 5150   CALL ERRMSG( ferr_invalid_command, status,
     .		'2D T-F times include missing value', *5000)
! 5200   CALL ERRMSG( ferr_invalid_command, status,
!     .		'Lag axis of F axis is not an integer number of time steps',
!     .		*5000)
	END
