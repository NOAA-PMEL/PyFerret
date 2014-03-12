	SUBROUTINE AUX_REGRID_LIMS_1D( dst_cx, src_cx, axis_list, naxis,
     .			               aux1, mr1, cx1, status )

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

*V690 3/13-1/14 *sh*

	IMPLICIT NONE

	include 'tmap_dims.parm'
	include 'implicit.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	src_cx, axis, dst_cx, naxis, axis_list(naxis), status,
     .		mr1, cx1, mr2, cx2, mr3, cx3 
        REAL*8  aux1(m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )

* internal variable declarations:
	LOGICAL increasing, increasing_here, looking_for_first
	INTEGER	KNOWN_GRID, 
     .		idim, src_grid, i, j, k, l, m, n, trans,
     .		kdstmin, kdstmax, kkmin, kkmax, klo, khi, kdel
	REAL*8  TM_WORLD, axmax, axmin, v, v1, bad_aux

* initialize
	bad_aux = mr_bad_data( mr1 )

* insert source grid (the defining grid for this variable)
* TEMPORARY -- ?move this block of code to AUX_VAR_REGRID when 2D curv is added?
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
	trans = cx_regrid_trans(z_dim,src_cx)    !  TEMPORARY: z_dim HARD CODED!!!

* increasing or decreasing? determine by finding and examining
* the first two valid values encountered in any column of the indicated axis
        DO 110 n = cx_lo_s6(cx1), cx_hi_s6(cx1)
        DO 110 m = cx_lo_s5(cx1), cx_hi_s5(cx1)
        DO 110 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
        DO 110 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
        DO 110 i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	   looking_for_first = .TRUE.
           DO 105 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	      v = aux1(i,j,k,l,m,n)
	      IF (v .EQ. bad_aux) THEN
	         GO TO 105
	      ELSEIF (looking_for_first) THEN
	         v1 = v
	         looking_for_first = .FALSE.
	      ELSE
	         increasing = v .GT. v1
	         GOTO 112
	      ENDIF
 105	   CONTINUE
 110	CONTINUE
* ... no line of this data had 2 valid values!  But continue anyway

* make sure that the auxiliary variable is monotonic
* note:  missing values in the array are allowed, but only at the edges
*        If a missing value is encountered in the body of the auxiliary var
*        regridding will not proceed beyong that subscript in the regrid line
 112	DO 120 n = cx_lo_s6(cx1), cx_hi_s6(cx1)
        DO 120 m = cx_lo_s5(cx1), cx_hi_s5(cx1)
        DO 120 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
        DO 120 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
        DO 120 i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	   looking_for_first = .TRUE.
           DO 115 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	      v = aux1(i,j,k,l,m,n)
	      IF (v .EQ. bad_aux) THEN
	         IF (looking_for_first) GOTO 115
	      ELSEIF (looking_for_first) THEN
	         v1 = v
	         looking_for_first = .FALSE.
	      ELSE
	         increasing_here = v .GT. v1
	         IF (increasing .NEQV. increasing_here) GOTO 5100  ! not monot.
	         v1 = v
	      ENDIF
 115	   CONTINUE
 120	CONTINUE


***** determine the minimal index limits needed to do the regrid *****
* SINGLE AXIS CODE ONLY (GET_CONTEXT_MODS setup to dis-allow multiple args for now)
* ... what are the desired world coordinate limts?
 130	idim    = axis_list(1)	! since this is 1D regridding
	kdstmin = cx_lo_ss ( dst_cx, idim )
	kdstmax = cx_hi_ss ( dst_cx, idim )
	IF (trans .EQ. pauxrgrd_linear 
     . .OR. trans .EQ. unspecified_int4  ) THEN
* ... linear interp requires source data below/above target grid point coordinate
* ... trans.EQ.unspec is for 3-argument plotting
	   axmin   = TM_WORLD(kdstmin, cx_grid(dst_cx), idim, box_middle)
	   axmax   = TM_WORLD(kdstmax, cx_grid(dst_cx), idim, box_middle)
	ELSEIF (trans .EQ. pauxrgrd_pwlave ) THEN
* ... averaging over cell requires source data over entire cell (larger region)
	   axmin   = TM_WORLD(kdstmin, cx_grid(dst_cx), idim, box_lo_lim)
	   axmax   = TM_WORLD(kdstmax, cx_grid(dst_cx), idim, box_hi_lim)
	ELSE
	   CALL ERRMSG( ferr_internal, status, 'unkwn rgrd trans', *5000)
	ENDIF
* when searching for the lo/hi index limts of the source data needed
* reverse the direction of traversing the array for decreasing-ordered data
* for a decreasing aux array min>max, lo>hi at this stage
	IF (increasing) THEN
	   klo   = cx_lo_s3(cx1)
	   khi   = cx_hi_s3(cx1)
	   kdel  = 1
	ELSE
	   khi  = cx_lo_s3(cx1)
	   klo  = cx_hi_s3(cx1)
	   kdel = -1
	ENDIF
	kkmin = khi
	kkmax = klo 

* find the smallest index range in the aux array that contains all of the aux
* values lying within the axmin to axmax value range on the destination axis 
* ==> CODE ONLY WORKS FOR Z AXIS               AT THIS STAGE !!!

        DO 290 n = cx_lo_s6(cx1), cx_hi_s6(cx1)
        DO 290 m = cx_lo_s5(cx1), cx_hi_s5(cx1)
        DO 290 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
        DO 290 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
        DO 290 i = cx_lo_s1(cx1), cx_hi_s1(cx1)

* ... search from low values to high for the upper index limit 
           DO 210 k = klo, khi, kdel
	      v = aux1(i,j,k,l,m,n)
	      IF ( v .EQ. bad_aux ) THEN
	         GOTO 210
	      ELSEIF (v .LT. axmax) THEN
	         IF ( increasing ) THEN
	            IF (kkmax .LT. k) kkmax = k
	         ELSE
	            IF (kkmax .GT. k) kkmax = k
	         ENDIF
	      ELSE
	         GOTO 220
	      ENDIF
 210       CONTINUE

* ... search from high values to low for the lower index limit 
 220	   DO 230 k = khi, klo, -1*kdel
	      v = aux1(i,j,k,l,m,n)
	      IF ( v .EQ. bad_aux ) THEN
	         GOTO 230
	      ELSEIF (v .GT. axmin) THEN
	         IF (increasing) THEN
	            IF (kkmin .GT. k) kkmin = k
	         ELSE
	            IF (kkmin .LT. k) kkmin = k
	         ENDIF
	      ELSE
	         GOTO 290
	      ENDIF
 230	   CONTINUE

 290    CONTINUE

* for decreasing-ordered aux array swap the lo/hi limits just found
	IF (.NOT.increasing) THEN
	   k     = kkmin
	   kkmin = kkmax
	   kkmax = k
	ENDIF

* round out, so that regridding interpolations will be possible at the edges
* (the value tests above were based on strict inequality so we always do this) 
	IF (kkmin .GT. cx_lo_ss(cx1,idim) ) kkmin = kkmin - 1
	IF (kkmax .LT. cx_hi_ss(cx1,idim) ) kkmax = kkmax + 1

        cx_lo_ss ( src_cx, idim )  = kkmin
        cx_hi_ss ( src_cx, idim )  = kkmax
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
 5100   CALL ERRMSG( ferr_invalid_command, status,
     .		'auxiliary regridding coordinates are not monotonic', *5000)
 5200   CALL ERRMSG( ferr_invalid_command, status,
     .		'auxiliary regridding coordinate field includes '//
     .		'missing value(s)', *5000)
	END
