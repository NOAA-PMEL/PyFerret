	SUBROUTINE DO_NEAREST(	idim, by_index, find_above,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx )

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
* that is nearest to the point

* This function performs 4 transformations
*       @CDA - closest distance above (world coord distance)
*       @CDB - closest distance below (world coord distance)
*       @CDA - closest index    above
*       @CDA - closest index    below

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 8/97
* v491 - *kob* 3.98 - modify loop @450 to use cx_lo/hi_s3 for k 
*                     rather than s4 *kob*

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	LOGICAL by_index, find_above
	INTEGER	idim, action, com_mr, com_cx, res_mr, res_cx
	REAL    com ( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res ( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	INTEGER i, j, k, l, grid, uplim, dnlim, offlim, delta,
     .		las_index
	REAL	bad_com, bad_res
	REAL*8  TM_WORLD

* --- end of introductory code ---

* initialize
	grid = mr_grid( com_mr )
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )

* determine axis limits along axis of transformation
	IF ( find_above ) THEN		     ! @NRDA and @NRIA
	  offlim = cx_hi_ss(com_cx, idim)    ! source data beyond edge
	  uplim = cx_lo_ss(res_cx, idim)
	  dnlim = cx_hi_ss(res_cx, idim)
	  delta = -1
	ELSE				     ! @NRDB and @NRIB
	  offlim = cx_lo_ss(com_cx, idim)    ! source data beyond edge
	  uplim = cx_hi_ss(res_cx, idim)
	  dnlim = cx_lo_ss(res_cx, idim)
	  delta =  1
	ENDIF

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 150 l = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	   DO 150 k = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	   DO 150 j = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
* ... locate first valid index off edge of requested region
	     las_index = unspecified_int4
	     DO 110 i = offlim, dnlim, delta
	       IF ( com(i,j,k,l) .NE. bad_com ) las_index = i
 110	     CONTINUE
* ... loop through points finding distance from last valid point
	      DO 120 i = dnlim, uplim, delta
	        IF ( com(i,j,k,l) .NE. bad_com ) THEN
	          las_index = i
	          res(i,j,k,l) = 0.0
	        ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	          res(i,j,k,l) = bad_res
	        ELSE
	          IF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l) = ABS(i-las_index)
	          ELSE
* ... distance in world coordinates
	            res(i,j,k,l) = ABS(
     .			TM_WORLD(i,         grid, idim, box_middle)
     .		      - TM_WORLD(las_index, grid, idim, box_middle) )
	          ENDIF
	       ENDIF
 120	     CONTINUE
 150	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 250 l = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	   DO 250 k = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	   DO 250 i = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
* ... locate first valid index off edge of requested region
	     las_index = unspecified_int4
	     DO 210 j = offlim, dnlim, delta
	       IF ( com(i,j,k,l) .NE. bad_com ) las_index = j
 210	     CONTINUE
* ... loop through points finding distance from last valid point
	      DO 220 j = dnlim, uplim, delta
	        IF ( com(i,j,k,l) .NE. bad_com ) THEN
	          las_index = j
	          res(i,j,k,l) = 0.0
	        ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	          res(i,j,k,l) = bad_res
	        ELSE
	          IF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l) = ABS(j-las_index)
	          ELSE
* ... distance in world coordinates
	            res(i,j,k,l) = ABS(
     .			TM_WORLD(j,         grid, idim, box_middle)
     .		      - TM_WORLD(las_index, grid, idim, box_middle) )
	          ENDIF
	       ENDIF
 220	     CONTINUE
 250	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 350 l = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	   DO 350 j = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
	   DO 350 i = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
* ... locate first valid index off edge of requested region
	     las_index = unspecified_int4
	     DO 310 k = offlim, dnlim, delta
	       IF ( com(i,j,k,l) .NE. bad_com ) las_index = k
 310	     CONTINUE
* ... loop through points finding distance from last valid point
	      DO 320 k = dnlim, uplim, delta
	        IF ( com(i,j,k,l) .NE. bad_com ) THEN
	          las_index = k
	          res(i,j,k,l) = 0.0
	        ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	          res(i,j,k,l) = bad_res
	        ELSE
	          IF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l) = ABS(k-las_index)
	          ELSE
* ... distance in world coordinates
	            res(i,j,k,l) = ABS(
     .			TM_WORLD(k,         grid, idim, box_middle)
     .		      - TM_WORLD(las_index, grid, idim, box_middle) )
	          ENDIF
	       ENDIF
 320	     CONTINUE
 350	   CONTINUE

* ALONG T AXIS
	ELSE

* i believe the k loop below should use cx_hi/lo_s3 rather than s4
* *kob*
	   DO 450 k = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	   DO 450 j = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
	   DO 450 i = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
* ... locate first valid index off edge of requested region
	     las_index = unspecified_int4
	     DO 410 l = offlim, dnlim, delta
	       IF ( com(i,j,k,l) .NE. bad_com ) las_index = l
 410	     CONTINUE
* ... loop through points finding distance from last valid point
	      DO 420 l = dnlim, uplim, delta
	        IF ( com(i,j,k,l) .NE. bad_com ) THEN
	          las_index = l
	          res(i,j,k,l) = 0.0
	        ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	          res(i,j,k,l) = bad_res
	        ELSE
	          IF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l) = ABS(l-las_index)
	          ELSE
* ... distance in world coordinates
	            res(i,j,k,l) = ABS(
     .			TM_WORLD(l,         grid, idim, box_middle)
     .		      - TM_WORLD(las_index, grid, idim, box_middle) )
	          ENDIF
	       ENDIF
 420	     CONTINUE
 450	   CONTINUE

	ENDIF

	RETURN
	END	


