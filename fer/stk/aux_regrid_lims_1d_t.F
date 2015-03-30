	SUBROUTINE AUX_REGRID_LIMS_1D_T( dst_cx, src_cx, axis_list, naxis,
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
* This routine handles the special case of forecast model  aggregations
* It determines the range of souce data on the T axis that is needed to
* regrid (reorganize) the 2D time array into a 1D time
* (orthogonal F and T axes) form

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

*v695+ 3/15 *sh* patterned after AUX_REGRID_LIMS_1D_Z

	IMPLICIT NONE

	include 'tmap_dims.parm'
	include 'implicit.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xtm_grid.cmn_text'    ! to sanity check T axis


* calling argument declarations:
	INTEGER	src_cx, axis, dst_cx, naxis, axis_list(naxis), status,
     .		mr1, cx1, mr2, cx2, mr3, cx3
        REAL*8  aux1(m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )

* internal variable declarations:
	INTEGER	KNOWN_GRID, CGRID_AXIS,
     .		idim, src_grid, i, j, k, l, m, n, trans,
     .		ndim, dim(nferdims), t_axis,
     .		ldstmin, ldstmax, llmin, llmax, llo, lhi, ldel,
     .		nlo, nhi, nmin, nmax, looking_for_first
	REAL*8  TM_WORLD, axmax, axmin, bad_aux, t, t1, delta,
     .		tdstrt, tdend, tstepmin, tstepmax

* initialize
	bad_aux = mr_bad_data( mr1 )
!	idim     = axis_list(1)
	llo      = cx_lo_s4(cx1)   ! 2d time array limit - should always be 1
	lhi      = cx_hi_s4(cx1)   ! should always be N
	nlo      = cx_lo_s6(cx1)   ! lowest forecast index - specified by user
	nhi      = cx_hi_s6(cx1)   ! highest forecast index - specified by user
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

* sanity check the target T axis
	t_axis = CGRID_AXIS(t_dim,dst_cx)
	IF (.NOT. line_regular(t_axis)) GOTO 5050
	IF (lhi .GT. llo) THEN
	   IF ( line_delta(t_axis)
     .	   .NE.(aux1(i,j,k,llo+1,m,nlo)-aux1(i,j,k,llo,m,nlo))) GOTO 5060
	ENDIF

* ideally we should check that the units of the 2D time field are the same as the units of the
* 1D time axis, but since typically Ferret code is creating the 1d time, we'll allow laziness

* this routine only for t_dim -- index-based remapping of time locations
	trans = cx_regrid_trans(t_dim,src_cx)
	IF (trans .NE. pauxrgrd_fmrc) 
     .	  CALL ERRMSG( ferr_internal, status, 'unkwn rgrd trans', *5000)

* make sure that the auxiliary variable is monotonic and regularly spaced
*	The 2D time array can never theoretically be non-monotonic
*       but FT aggregations are new and unreliable
	DO n = nlo, nhi
	   looking_for_first = 0
           DO l = llo, lhi
	      t = aux1(i,j,k,l,m,n)
	      IF (t .EQ. bad_aux) THEN
	         GOTO 5150
	      ELSEIF (looking_for_first.EQ.0) THEN
	         t1 = t
	         looking_for_first = 1
	      ELSE
	         IF (looking_for_first.EQ.1) THEN
	            delta = t-t1
	            IF (delta .LT. 0.0) GOTO 5100
	            looking_for_first = 2
                 ENDIF
	         IF (delta .NE. (t-t1)) GOTO 5100
	         t1 = t
	      ENDIF
	   ENDDO
	ENDDO

* the relationship between the time steps in the 2D aux time field (cx1)
* and the same times in the destination field looks something
* like this over 3 forecast cycles

*   **     2d aux times    **	      ** orthoganalized T-F field **
*								115
*                       		               1013	114
*                       		        1012   1012	113
*                       		        1011   1011	112
*                       		       >1010   1010	111
*                       		        1009   1009	110
*   18     1008   1012   1016		 1008   1008   1008	109
*   17     1007   1011   1015		 1007   1007		108
*   16     1006  >1010   1014		 1006   1006		107
*   15     1005   1009   1013		 1005  >1005		106
*   14     1004   1008   1012		 1004   1004		105
*   13     1003   1007   1011		 1003			104
*   12     1002   1006   1010		 1002			103
*   11     1001  >1005   1009		 1001			102
*   10     1000   1004   1008		 1000			101
*    |								  |
* l in src_cx						    l in dst_cx 

* say that the requested time range is T=1005:1010
* The lowest L index of source data needed is the index at which the START time
*    is first encountered as we loop down the forecasts searching lo to hi
* The highest L index of source data needed is the index at which the END  time
*    is first encountered as we loop up   the forecasts searching hi to lo


* This commented out block of code increases efficiency by requesting only
* only the F-range that contains data relevant to the requested T-range.
* However, doing so causes the regrid operation to alter the N range.
* Since this is a 1D T regrid operation, "var[GT(time2d)=t1d]", it is not
* appropriate to alter the F range.  The efficiency loss effects only a minor
* situation in which for whatever reason the user has requested N-indices of
* pure missing data.
!* bail out if the requested min exceeds the available range or similar max
!	IF (tdstrt .GT. tstepmax
!     .  .OR. tdend .LT. tstepmin ) THEN
!* ... no regridding is possible - result will be a field of missing data
!* ... provide a nominal data region that is not trouble to get
!	   llmax = llmin  ! request a single point region only of source data
!	   nmax  = nmin
!	   GOTO 500
!	ENDIF
!
!* determine the minimal forecast index range (i.e. with relevant timesteps)
!	DO n = nlo, nhi
!	   t  = aux1(i,j,k,llo,m,n)
!	   t1 = aux1(i,j,k,lhi,m,n)
!	   IF (t1 .GE. tdstrt
!     .   .AND. t  .LE. tdend  ) THEN
!* ... found first forecast with relevant time steps
!	      nmin = n
!	      EXIT
!	   ENDIF
!	ENDDO
!	DO n = nhi, nlo, -1
!	   t  = aux1(i,j,k,llo,m,n)
!	   t1 = aux1(i,j,k,lhi,m,n)
!	   IF (t1 .GE. tdstrt
!     .   .AND. t  .LE. tdend  ) THEN
!* ... found last forecast with relevant time steps
!	      nmax = n
!	      EXIT
!	   ENDIF
!	ENDDO
!	IF (nmin .GT. nmax) nmax = nmin


***** determine the minimal src time index range needed to do the regrid *****

* find the lowest index of source data that needs to be requested
	IF (tdstrt .LT. tstepmin) THEN
	   CONTINUE	   ! keep llmin
	ELSE
* ... search aux times from low values to high n index for the start time
	   n = nmax + 1
	   DO WHILE (n > nmin)
              n = n - 1           ! loop *down* F indices
	      DO l = llo, lhi     ! loop *up*   T indices
	         t = aux1(i,j,k,l,m,n)
	         IF (t .GT. tdstrt) THEN
	            EXIT    ! continue search at next lower forcast
	         ELSEIF (t .EQ. tdstrt) THEN
	            llmin = l
	            n = nlo  ! signal termination of F loop 
	            EXIT
	         ENDIF
	      ENDDO
	   ENDDO
	ENDIF

* find the highest index of source data that needs to be requested
	IF (tdend .GT. tstepmax) THEN
	   CONTINUE	   ! keep llmax
	ELSE

* ... search aux times from high values to low for the end time
	   n = nmin-1
	   DO WHILE (n < nmax)
	      n = n + 1            ! loop *up*   F indices
	      DO l = lhi, llo, -1  ! loop *down* T indices
	         t = aux1(i,j,k,l,m,n)
	         IF (t .LT. tdend) THEN
	            EXIT    ! continue search at next higher forcast
	         ELSEIF (t .EQ. tdend) THEN
	            llmax = l
	            n = nhi ! signal termination of F loop
	            EXIT 
	         ENDIF
	      ENDDO
	   ENDDO
	ENDIF

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
 5050   CALL ERRMSG( ferr_invalid_command, status,
     .		'target 1D time axis is not regular', *5000)
 5060   CALL ERRMSG( ferr_invalid_command, status,
     .		'delta T of target 1D time axis doesnt match the data', *5000)
 5100   CALL ERRMSG( ferr_invalid_command, status,
     .		'2D T-F times are not increasing and regular', *5000)
 5150   CALL ERRMSG( ferr_invalid_command, status,
     .		'2D T-F times include missing value', *5000)
	END
