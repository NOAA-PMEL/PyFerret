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

*V685 3/13 *sh*

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
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          aux2(m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .          aux3(m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,
     .               m3lot:m3hit,m3loe:m3hie,m3lof:m3hif )

* internal variable declarations:
	LOGICAL increasing
	INTEGER	KNOWN_GRID, 
     .		idim, src_grid, i, j, k, l, m, n, kkmin, kkmax
	REAL*8  TM_WORLD, axmax, axmin, v, v1, bad_aux

* initialize
	bad_aux = mr_bad_data( mr1 )

* insert source grid (the defining grid for this variable)
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
 100    CONTINUE

***** determine the minimal index limits needed to do the regrid *****
* SINGLE AXIS CODE ONLY (GET_CONTEXT_MODS setup to dis-allow multiple args for now)
* ... what are the desired world coordinate limts?
	idim = axis_list(1)    ! TEMPORARY CODE FINISH ME !!!!! XXXX
	kkmin = cx_hi_ss ( dst_cx, idim )
	kkmax = cx_lo_ss ( dst_cx, idim )
	axmax = TM_WORLD(kkmin, cx_grid(dst_cx), idim, box_middle)
	axmin = TM_WORLD(kkmax, cx_grid(dst_cx), idim, box_middle)

* search through aux1 for the lowest and highest indices needed
* ==> CODE ONLY WORKS FOR MONOTONIC INCREASING AT THIS STAGE !!!

        DO 210 n = cx_lo_s6(cx1), cx_hi_s6(cx1)
        DO 210 m = cx_lo_s5(cx1), cx_hi_s5(cx1)
        DO 210 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
        DO 210 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
        DO 210 i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	   v1 = aux1(i,j,cx_lo_s3(cx1),l,m,n)
	   increasing = aux1(i,j,cx_hi_s3(cx1),l,m,n) .GT. v1

* ... first search in increasing order for the largest index containing a
*     value smaller than axmax, the requested max coordinate value
           DO 200 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	      v = aux1(i,j,k,l,m,n)
	      IF ( v .EQ. bad_aux ) THEN
	         GOTO 5200
* ... check monotonicity
	      ELSEIF (k .GT. cx_lo_s3(cx1)) THEN
	         IF (increasing) THEN
	            IF (v .LE. v1) GOTO 5100
	         ELSE
	            IF (v .GE. v1) GOTO 5100
	         ENDIF
	      ENDIF

* ... point lies within desired bounds? shift max index if necessary
	      IF (v .LT. axmax) THEN
	         IF (kkmax .LT. k) kkmax = k
	      ENDIF
 200       CONTINUE


* ... second search in decreasing order for the smallest index containing a
*     value larger than axmin, the requested min coordinate value
           DO 202 k = cx_hi_s3(cx1), cx_lo_s3(cx1), -1
	      v = aux1(i,j,k,l,m,n)

* ... point lies within desired bounds? shift min index if necessary
	      IF (v .GT. axmin) THEN
	         IF (kkmin .GT. k) kkmin = k
	      ENDIF
 202       CONTINUE

 210    CONTINUE

* round out, so that regridding interpolations will be possible at the edges
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
