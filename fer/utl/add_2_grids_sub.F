	SUBROUTINE ADD_2_GRIDS_SUB (
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit, bad_src,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit, bad_dst )

* add data from one grid to another

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

	   IF (  src(i,j,k,l) .NE. bad_src
     .	   .AND. dst  (i,j,k,l) .NE. bad_dst ) THEN

	      dst(i,j,k,l) = dst(i,j,k,l) + src(i,j,k,l)

	   ELSE

	      dst(i,j,k,l) = bad_dst

	   ENDIF

 100	CONTINUE

	RETURN
	END 
