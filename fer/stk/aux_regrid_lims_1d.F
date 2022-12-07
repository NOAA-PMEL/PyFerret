	SUBROUTINE AUX_REGRID_LIMS_1D( dst_cx, src_cx,
     .				       the_axis, axloss, axhiss,
     .			               aux, aux_mr, aux_cx, status )

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
* Given a context describing data to be regridded to a destination grid
* create the description of the needed source data in the source context
* Use the auxiliary variables to determine the index range of data needed
* For example, if the regrid has requested depth from 0 to 100 and in the 
* auxiliary variable containing depths all of the values LE 100
* are in K=1:6 then we need only request source data over the range K=1:7 

* When called the source context should be a copy of the destination context 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

*v695+ 3/15 *sh* from AUX_REGRID_LIMS_1D_Z
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

	IMPLICIT NONE

	include 'tmap_dims.parm'
	include 'implicit.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	src_cx, axis, dst_cx, the_axis, axloss, axhiss, status,
     .		aux_mr, aux_cx
        REAL*8  aux(m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .              m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )
	REAL*8 line(axloss:axhiss)

* internal variable declarations:
	LOGICAL increasing, looking_for_first
	INTEGER	KNOWN_GRID, 
     .		idim, src_grid, i, j, k, l, m, n, trans,
     .		kdstmin, kdstmax, ss, ssmin, ssmax, klo, khi, kdel,
     .		ss_lo(nferdims), ss_hi(nferdims), n_incr, n_decr
	REAL*8  TM_WORLD, axmax, axmin, v, v1, bad_aux

* initialize
	bad_aux = mr_bad_data( aux_mr )
	trans = cx_regrid_trans(the_axis,src_cx)

* capture the data range and modify the target axis to a single point
	DO idim = 1, nferdims
	  ss_lo(idim) = cx_lo_ss(aux_cx, idim)
	  ss_hi(idim) = cx_hi_ss(aux_cx, idim)
	ENDDO
	axloss = ss_lo(the_axis)
	axhiss = ss_hi(the_axis)
* ... a dummy loop on the designated axis
	ss_lo(the_axis) = 1
	ss_hi(the_axis) = 1

* insert source grid (the defining grid for this variable)
* XXXX TEMPORARY -- ?move this block of code to AUX_VAR_REGRID when 2D curv is added?
	src_grid = KNOWN_GRID( cx_data_set(src_cx),
     .			       cx_category(src_cx),
     .			       cx_variable(src_cx)  )
	cx_grid( src_cx ) = src_grid
	cx_unstand_grid( src_cx ) = .FALSE.
	is_phase( isp )   = prgrd_from_source
	cx_naux( src_cx ) = 0
        DO idim = 1, nferdims
           cx_aux_var  (idim, src_cx) = unspecified_int4
           cx_aux_cat  (idim, src_cx) = unspecified_int4
           cx_aux_stat (idim, src_cx) = paux_stat_na
	ENDDO

* Weak test to determine increasing or decreasing by finding the first column
* that has a clear order.   NOTE:  This test can give wrong answers on
* pathological fields with differing order columns
	
	DO n = ss_lo(6),ss_hi(6)
	DO m = ss_lo(5),ss_hi(5)
	DO l = ss_lo(4),ss_hi(4)
	DO k = ss_lo(3),ss_hi(3)
	DO j = ss_lo(2),ss_hi(2)
	DO i = ss_lo(1),ss_hi(1)

* ... get a column of data
	   CALL EXTRACT_ARG1_LINE( aux, the_axis, axloss, axhiss,
     .				   i, j, k, l, m, n, line )

	   n_incr = 0
	   n_decr = 0
	   looking_for_first = .TRUE.
           DO ss = axloss, axhiss
	      v = line(ss)
	      IF (v .EQ. bad_aux) THEN
	         CYCLE
	      ELSEIF (looking_for_first) THEN
	         v1 = v
	         looking_for_first = .FALSE.
	      ELSE
	         IF (v .GT. v1) THEN
	            n_incr = n_incr + 1
	         ELSE
	            n_decr = n_decr + 1
	         ENDIF
	         v1 = v
	      ENDIF
	   ENDDO

* ... is there a clear ordering to this column of auxilinary coordinates?
	   IF ((n_incr.GE.1 .OR. n_decr.GE.1)
     .   .AND. (n_incr .NE. n_decr          )  ) THEN
	    increasing = n_incr .GE. n_decr  
	    GOTO 130
	   ENDIF

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

***** determine the minimal index limits needed to do the regrid *****
* SINGLE AXIS CODE ONLY (GET_CONTEXT_MODS setup to dis-allow multiple args for now)
* ... what are the desired world coordinate limts?
 130	kdstmin = cx_lo_ss ( dst_cx, the_axis )
	kdstmax = cx_hi_ss ( dst_cx, the_axis )
	IF (cx_aux_stat(the_axis,dst_cx) .EQ. paux_stat_false_1pt_regrid) THEN
* ... false 1pt regridding (no target axis ... just a single point value)
	   axmin = cx_lo_ww(the_axis, dst_cx)
	   axmax = cx_hi_ww(the_axis, dst_cx)
	ELSEIF (trans .EQ. pauxrgrd_linear 
     .     .OR. trans .EQ. unspecified_int4  ) THEN
* ... linear interp requires source data below/above target grid point coord
* ... trans.EQ.unspec is for 3-argument plotting
	   axmin   = TM_WORLD(kdstmin, cx_grid(dst_cx), the_axis, box_middle)
	   axmax   = TM_WORLD(kdstmax, cx_grid(dst_cx), the_axis, box_middle)
	ELSEIF (trans .EQ. pauxrgrd_pwlave ) THEN
* ... averaging over cell requires source data over entire cell (larger region)
	   axmin   = TM_WORLD(kdstmin, cx_grid(dst_cx), the_axis, box_lo_lim)
	   axmax   = TM_WORLD(kdstmax, cx_grid(dst_cx), the_axis, box_hi_lim)
	ELSE
	   CALL ERRMSG( ferr_internal, status, 'unkwn rgrd trans', *5000)
	ENDIF
* when searching for the lo/hi index limts of the source data needed
* reverse the direction of traversing the array for decreasing-ordered data
* for a decreasing aux array min>max, lo>hi at this stage
	IF (increasing) THEN
	   klo   = cx_lo_ss(aux_cx,the_axis)
	   khi   = cx_hi_ss(aux_cx,the_axis)
	   kdel  = 1
	ELSE
	   khi  = cx_lo_ss(aux_cx,the_axis)
	   klo  = cx_hi_ss(aux_cx,the_axis)
	   kdel = -1
	ENDIF
	ssmin = khi
	ssmax = klo 

* find the smallest index range in the aux array that contains all of the aux
* values lying within the axmin to axmax value range on the destination axis 
	DO n = ss_lo(6),ss_hi(6)
	DO m = ss_lo(5),ss_hi(5)
	DO l = ss_lo(4),ss_hi(4)
	DO k = ss_lo(3),ss_hi(3)
	DO j = ss_lo(2),ss_hi(2)
	DO i = ss_lo(1),ss_hi(1)

* get a column of data
	   CALL EXTRACT_ARG1_LINE( aux, the_axis, axloss, axhiss,
     .				   i, j, k, l, m, n, line )

* ... search from low values to high for the upper index limit 
           DO ss = klo, khi, kdel
	      v = line(ss)
	      IF ( v .EQ. bad_aux ) THEN
	         CYCLE
	      ELSEIF (v .LT. axmax) THEN
	         IF ( increasing ) THEN
	            IF (ssmax .LT. ss) ssmax = ss
	         ELSE
	            IF (ssmax .GT. ss) ssmax = ss
	         ENDIF
	      ELSE
	         EXIT
	      ENDIF
	   ENDDO

* ... search from high values to low for the lower index limit 
	   DO ss = khi, klo, -1*kdel
	      v = line(ss)
	      IF ( v .EQ. bad_aux ) THEN
	         CYCLE
	      ELSEIF (v .GT. axmin) THEN
	         IF (increasing) THEN
	            IF (ssmin .GT. ss) ssmin = ss
	         ELSE
	            IF (ssmin .LT. ss) ssmin = ss
	         ENDIF
	      ELSE
	         GOTO 290
	      ENDIF
	   ENDDO

 290       CONTINUE

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

* for decreasing-ordered aux array swap the lo/hi limits just found
	IF (.NOT.increasing) THEN
	   ss    = ssmin
	   ssmin = ssmax
	   ssmax = ss
	ENDIF

* round out, so that regridding interpolations will be possible at the edges
* (the value tests above were based on strict inequality so we always do this) 
	IF (ssmin .GT. cx_lo_ss(aux_cx,the_axis) ) ssmin = ssmin - 1
	IF (ssmax .LT. cx_hi_ss(aux_cx,the_axis) ) ssmax = ssmax + 1

        cx_lo_ss ( src_cx, the_axis )  = ssmin
        cx_hi_ss ( src_cx, the_axis )  = ssmax
        cx_by_ss ( the_axis, src_cx )  = .TRUE.

        cx_lo_ww ( the_axis, src_cx )  = unspecified_val8
        cx_hi_ww ( the_axis, src_cx )  = unspecified_val8
        cx_delta ( the_axis, src_cx )  = unspecified_val8
        cx_trans ( the_axis, src_cx )  = trans_no_transform
        cx_trans_arg(the_axis,src_cx)  = bad_val4
        cx_given( the_axis, src_cx )  = .FALSE.
        cx_regrid_trans(the_axis,src_cx) = unspecified_int4 
        cx_formatted( the_axis )   = .FALSE.

	CALL FLESH_OUT_AXIS( the_axis, src_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* success
        status = ferr_ok
	RETURN

* error exit
 5000	RETURN
! 5100   CALL ERRMSG( ferr_invalid_command, status,
!     .		'auxiliary regridding coordinates are too disordered'
!     .          //pCR//'Need 2:1 ratio ordered:disordered', *5000)
	END
