	SUBROUTINE COPY_GRID_SUB (
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit, bad_src,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit, bad_dst )

* support routine for COPY_GRID with unsubscripted limits

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
	INTEGER slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit )
	REAL	bad_src, bad_dst

* internal variable declarations:
	INTEGER	i, j, k, l

* --- end of introductory code ---

	DO 100 l = dlot, dhit
	DO 100 k = dloz, dhiz
	DO 100 j = dloy, dhiy
	DO 100 i = dlox, dhix

	   dst(i,j,k,l) = src(i,j,k,l)

 100	CONTINUE

* check for differing bad data flags in an efficient way ...
	IF ( bad_src .NE. bad_dst ) THEN
	   DO 200 l = dlot, dhit
	   DO 200 k = dloz, dhiz
	   DO 200 j = dloy, dhiy
	   DO 200 i = dlox, dhix

	      IF ( src(i,j,k,l) .EQ. bad_src ) dst(i,j,k,l) = bad_dst

 200	   CONTINUE
	ENDIF

	RETURN
	END 
