	SUBROUTINE REVERSE_GRID_AXIS_SUB( axis, cx, dat,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit )

* copy data from source grid into dstination grid using limits of source
* context (i.e. dstination grid is ASSUMED) big enough to hold data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V320: 2/16/89

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER axis, cx,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit
	REAL    dat( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit )

* internal variable declarations:
	INTEGER	i, j, k, l, turn_pt, hiplo
	REAL tmp

* --- end of introductory code ---

* initialize
	turn_pt = cx_lo_ss(cx,axis)
     .	      + (cx_hi_ss(cx,axis)-cx_lo_ss(cx,axis)-1)/2
	hiplo = cx_lo_ss(cx,axis) + cx_hi_ss(cx,axis)

* reverse the data along the appropriate axis

	IF ( axis .EQ. x_dim ) THEN
	   DO 100 l = cx_lo_s4(cx), cx_hi_s4(cx)
	   DO 100 k = cx_lo_s3(cx), cx_hi_s3(cx)
	   DO 100 j = cx_lo_s2(cx), cx_hi_s2(cx)
	   DO 100 i = cx_lo_s1(cx), turn_pt

	      tmp = dat(i,j,k,l)
	      dat(      i,j,k,l) = dat(hiplo-i,j,k,l)
	      dat(hiplo-i,j,k,l) = tmp

 100	   CONTINUE

	ELSEIF ( axis .EQ. y_dim ) THEN
	   DO 200 l = cx_lo_s4(cx), cx_hi_s4(cx)
	   DO 200 k = cx_lo_s3(cx), cx_hi_s3(cx)
	   DO 200 j = cx_lo_s2(cx), turn_pt
	   DO 200 i = cx_lo_s1(cx), cx_hi_s1(cx)

	      tmp = dat(i,j,k,l)
	      dat(i,      j,k,l) = dat(i,hiplo-j,k,l)
	      dat(i,hiplo-j,k,l) = tmp

 200	   CONTINUE

	ELSEIF ( axis .EQ. z_dim ) THEN
	   DO 300 l = cx_lo_s4(cx), cx_hi_s4(cx)
	   DO 300 k = cx_lo_s3(cx), turn_pt
	   DO 300 j = cx_lo_s2(cx), cx_hi_s2(cx)
	   DO 300 i = cx_lo_s1(cx), cx_hi_s1(cx)

	      tmp = dat(i,j,k,l)
	      dat(i,j,      k,l) = dat(i,j,hiplo-k,l)
	      dat(i,j,hiplo-k,l) = tmp

 300	   CONTINUE

	ELSEIF ( axis .EQ. t_dim ) THEN
	   DO 400 l = cx_lo_s4(cx), turn_pt
	   DO 400 k = cx_lo_s3(cx), cx_hi_s3(cx)
	   DO 400 j = cx_lo_s2(cx), cx_hi_s2(cx)
	   DO 400 i = cx_lo_s1(cx), cx_hi_s1(cx)

	      tmp = dat(i,j,k,l)
	      dat(i,j,k,      l) = dat(i,j,k,hiplo-l)
	      dat(i,j,k,hiplo-l) = tmp

 400	   CONTINUE

	ENDIF

	RETURN
	END 


