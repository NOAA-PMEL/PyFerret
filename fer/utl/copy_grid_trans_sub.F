	SUBROUTINE COPY_GRID_TRANS_SUB( md1, md2,
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit, bad_src,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit, bad_dst )

* copy data from one grid to another transposing dim1 and dim2 components

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:	10/16/89 - from COPY_GRID_TRANS

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
* note: dst has dimensions transposed
	INTEGER md1, md2,
     .		slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit ),
     .		dst( md2, md1 )
	REAL	bad_src, bad_dst

* internal variable declarations:
	INTEGER	ii, jj, i, j, k, l

* --- end of introductory code ---

	ii = 1
	jj = 1
	DO 100 l = dlot, dhit
	DO 100 k = dloz, dhiz
	DO 100 j = dloy, dhiy
	DO 100 i = dlox, dhix

	   dst(jj,ii) = src(i,j,k,l)

	   ii = ii + 1
	   IF ( ii .GT. md1 ) THEN
	      ii = 1
	      jj = jj + 1
	   ENDIF

 100	CONTINUE

* check for differing bad data flags in an efficient way ...
	IF ( bad_src .NE. bad_dst ) THEN
	   DO 200 ii = 1, md1
	   DO 200 jj = 1, md2

	      IF ( dst(jj,ii) .EQ. bad_src ) dst(jj,ii) = bad_dst

 200	   CONTINUE
	ENDIF

	RETURN
	END 
