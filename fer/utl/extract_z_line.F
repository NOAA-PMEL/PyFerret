	SUBROUTINE EXTRACT_Z_LINE( block, bad_flag, i, j, l,
     .		cxloz, cxhiz,
     .		slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		zline, klo, khi )

* Extract a single Z line of data from a memory variable grid.
* Identify contiguously valid data region and report in klo, khi
* Return values in buffer that always starts with the first valid value

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450:  11/12/97
* V491:  16/4/98 - added cx_lo/hiz to limit extraction by context limits

* calling argument declarations:
	INTEGER slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		i, j, l, klo, khi, cxloz, cxhiz
	REAL    bad_flag,
     .		block( slox:shix,sloy:shiy,sloz:shiz,slot:shit ),
     .		zline(shiz-sloz+1)

* internal variable declarations:
	LOGICAL	searching_for_valid
	INTEGER k

* initialize
	searching_for_valid = .TRUE.

* initialize
	klo = 0		! if unchanged at end signals no valid data found
	khi = 0

* extract the line
	DO 100 k = cxloz, cxhiz

* ... skip invalid values at start of line
	   IF ( searching_for_valid ) THEN
	      searching_for_valid = block(i,j,k,l) .EQ. bad_flag
	      IF (searching_for_valid) GOTO 100
	      klo = k
* ... terminate if invalid point encountered after valid points
	   ELSEIF ( block(i,j,k,l) .EQ. bad_flag ) THEN
	      khi = k-1
	      GOTO 300
	   ENDIF

* ... transfer point to output
	   zline(k-klo+1) = block(i,j,k,l)

 100	CONTINUE
	khi = cxhiz

* successful completion
* reference returned limits to 1, so that klo=0 reliably signals no data
 300	klo = klo - cxloz + 1
	khi = khi - cxloz + 1
	RETURN
	END
