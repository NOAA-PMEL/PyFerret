	SUBROUTINE LIN_REGRID_1_AXIS(	cx_lims, axis, dst_lo, dst_hi,
     .					src, msrc, dst, mdst,
     .					ss12, coef )

* 	Using data values from src organized on grid src_grd, interpolate to
* 	values dst on grid dst_grd.

* 	msrc and mdst point to data structures in COMMON/XVARIABLES/ which
* 	specify the subscript bounds of src and dst within their respective
* 	grids.

* 	Both src and dst are assumed to be 4 dimensional structures
*	with regridding needed along axis "axis"

*	ss12 holds the indices of nearest grid nodes on src_grd for nodes
*	on dst_grd. T holds the interpolation coefficients.

*	The interpolation algorithm is taken from Numerical Recipes, Chapter 3,
*	Section "Interpolation in Two or More Dimensions"

* 	programmer - Jerry Davison
* 	NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program
* 	written for VAX computer under VMS operating system
* 	revision 1.0 - 5.2.88
* V205:   8/3/89 - major re-write (regrid along 1 axis only of 4D structure)
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* CALLING ARGUMENT DECLARATIONS:
	INTEGER	  cx_lims, axis, dst_lo, dst_hi, msrc, mdst
	INTEGER   ss12(	dst_lo:dst_hi )
	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		coef( dst_lo:dst_hi )

* INTERNAL VARIABLE DECLARATIONS:
	INTEGER i, j, k, l,
     .		ss, src_lo, src_hi, src_grd, dst_grd
	REAL	bad_src, bad_dst, z1, z2, c1, c2

********************************************************************************

* LIMITS FOR CALCULATION
        src_lo = mr_lo_ss(msrc,axis)
        src_hi = mr_hi_ss(msrc,axis)

* FLAG FOR BAD/MISSING DATA
        bad_src = mr_bad_data(msrc)
        bad_dst = mr_bad_data(mdst)

* source and destination grids
	src_grd = mr_grid( msrc )
	dst_grd = mr_grid( mdst )

* DETERMINE (LOWER) src_grd INDEX OF NEAREST NEIGHBORS ON src_grd AXES
* FOR NODES ON dst_grd AXES, AND COMPUTE FRACTIONAL POSITIONS OF dst_grd
* NODES WITH RESPECT TO src_grd.
	CALL GET_LINEAR_COEF(	src_lo, src_hi, src_grd,
     .				dst_lo, dst_hi, dst_grd,
     .				axis, ss12, coef )

* CONVERT THE DATA TO dst_grd
	IF ( axis .EQ. x_dim ) THEN
	   DO 110 i = dst_lo, dst_hi

* interpolation coefficients
	      ss = ss12(i)
	      c2 = coef(i)
	      c1 = 1.0 - c2

* special case: exact overlap of source and dest points
	      IF ( c1 .EQ. 1.0 ) THEN
	         DO 100 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 100 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 100 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
 100	         dst(i,j,k,l) = src(ss,j,k,l)
	         GOTO 110
	      ELSEIF ( ss .EQ. unspecified_int4 ) THEN
* ... this should never happen in FERRET 2.00+
	         DO 102 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 102 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 102 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
 102		 dst(i,j,k,l) = bad_dst
	         GOTO 110
	      ENDIF

* regular interpolation
	      DO 105 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	      DO 105 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	      DO 105 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)

* surrounding source data points
	         z1 = src(ss  ,j,k,l)
	         z2 = src(ss+1,j,k,l)

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src ) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1*z1 + c2*z2
	         ENDIF

 105	      CONTINUE
 110	   CONTINUE

	ELSEIF ( axis .EQ. y_dim ) THEN
	   DO 210 j = dst_lo, dst_hi

* interpolation coefficients
	      ss = ss12(j)
	      c2 = coef(j)
	      c1 = 1.0 - c2

* special case: exact overlap of source and dest points
	      IF ( c1 .EQ. 1.0 ) THEN
	         DO 200 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 200 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 200 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 200	         dst(i,j,k,l) = src(i,ss,k,l)
	         GOTO 210
	      ELSEIF ( ss .EQ. unspecified_int4 ) THEN
* ... this should never happen in FERRET 2.00+
	         DO 202 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 202 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 202 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 202		 dst(i,j,k,l) = bad_dst
	         GOTO 210
	      ENDIF

* regular interpolation
	      DO 205 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	      DO 205 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	      DO 205 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* surrounding source data points
	         z1 = src(i,ss  ,k,l)
	         z2 = src(i,ss+1,k,l)

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src ) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1*z1 + c2*z2
	         ENDIF

 205	      CONTINUE
 210	   CONTINUE

	ELSEIF ( axis .EQ. z_dim ) THEN
	   DO 310 k = dst_lo, dst_hi

* interpolation coefficients
	      ss = ss12(k)
	      c2 = coef(k)
	      c1 = 1.0 - c2

* special case: exact overlap of source and dest points
	      IF ( c1 .EQ. 1.0 ) THEN
	         DO 300 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 300 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO 300 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 300	         dst(i,j,k,l) = src(i,j,ss,l)
	         GOTO 310
	      ELSEIF ( ss .EQ. unspecified_int4 ) THEN
* ... this should never happen in FERRET 2.00+
	         DO 302 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 302 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO 302 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 302		 dst(i,j,k,l) = bad_dst
	         GOTO 310
	      ENDIF

* regular interpolation
	      DO 305 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	      DO 305 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	      DO 305 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* surrounding source data points
	         z1 = src(i,j,ss  ,l)
	         z2 = src(i,j,ss+1,l)

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src ) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1*z1 + c2*z2
	         ENDIF

 305	      CONTINUE
 310	   CONTINUE

	ELSEIF ( axis .EQ. t_dim ) THEN
	   DO 410 l = dst_lo, dst_hi

* interpolation coefficients
	      ss = ss12(l)
	      c2 = coef(l)
	      c1 = 1.0 - c2

* special case: exact overlap of source and dest points
	      IF ( c1 .EQ. 1.0 ) THEN
	         DO 400 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 400 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO 400 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 400	         dst(i,j,k,l) = src(i,j,k,ss)
	         GOTO 410
	      ELSEIF ( ss .EQ. unspecified_int4 ) THEN
* ... this should never happen in FERRET 2.00+
	         DO 402 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 402 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO 402 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 402		 dst(i,j,k,l) = bad_dst
	         GOTO 410
	      ENDIF

* regular interpolation
	      DO 405 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	      DO 405 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	      DO 405 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)

* surrounding source data points
	         z1 = src(i,j,k,ss  )
	         z2 = src(i,j,k,ss+1)

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src ) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1*z1 + c2*z2
	         ENDIF

 405	      CONTINUE
 410	   CONTINUE
	ENDIF

	RETURN
	END
