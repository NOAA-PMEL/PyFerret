	SUBROUTINE DO_NEAREST(	idim, by_index, find_above,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx )

* for each point, locate the valid point along the specified axis
* that is nearest to the point

* This function performs 4 transformations
*       @CDA - closest distance above (world coord distance)
*       @CDB - closest distance below (world coord distance)
*       @CDA - closest index    above
*       @CDA - closest index    below

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 8/97

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
	   DO 450 k = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
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


