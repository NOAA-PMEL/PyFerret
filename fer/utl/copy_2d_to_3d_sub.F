	SUBROUTINE COPY_2D_TO_3D_SUB (
     .		src, slox, shix, sloy, shiy,             slot, shit,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit )

* support routine for COPY_2D_TO_3D with unsubscripted limits

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/12/89

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER slox, shix, sloy, shiy,             slot, shit,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit
	REAL    src( slox:shix,sloy:shiy,          slot:shit ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit )

* internal variable declarations:
	INTEGER	i, j, k, l, k1

* --- end of introductory code ---

* copy one XY plane to the dstination grid
	k1 = dloz

	DO 100 l = dlot, dhit
	DO 100 j = dloy, dhiy
	DO 100 i = dlox, dhix

	   dst(i,j,k1,l) = src(i,j,l)

 100	CONTINUE

* (no check for bad data - same flag assumed on source and destination)

* now replicate along the Z axis
	DO 210 k = k1+1, dhiz

	   DO 200 l = dlot, dhit
	   DO 200 j = dloy, dhiy
	   DO 200 i = dlox, dhix
 200	   dst(i,j,k,l) = dst(i,j,k1,l)

 210	CONTINUE

	RETURN
	END 
