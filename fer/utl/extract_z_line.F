	SUBROUTINE EXTRACT_Z_LINE( block, bad_flag, i, j, l,
     .		slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		zline, klo, khi )

* Extract a single Z line of data from a memory variable grid.
* Identify contiguously valid data region and report in klo, khi

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450:  11/12/97

* calling argument declarations:
	INTEGER slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		i, j, l, klo, khi
	REAL    bad_flag,
     .		block( slox:shix,sloy:shiy,sloz:shiz,slot:shit ),
     .		zline(shiz-sloz+1)

* internal variable declarations:
	LOGICAL	searching_for_valid
	INTEGER k

* initialize
	searching_for_valid = .TRUE.

* initialize
	klo = 0
	khi = 0

* extract the line
	DO 100 k = sloz, shiz

* ... skip invalid values at start of line
	   IF ( searching_for_valid ) THEN
	      searching_for_valid = block(i,j,k,l) .EQ. bad_flag
	      IF (searching_for_valid) GOTO 100
	      klo = k-sloz+1
* ... terminate if invalid point encountered after valid points
	   ELSEIF ( block(i,j,k,l) .EQ. bad_flag ) THEN
	      khi = k-sloz
	      RETURN
	   ENDIF

* ... transfer point to output
	   zline(k-klo+1) = block(i,j,k,l)

 100	CONTINUE

	khi = shiz-sloz+1
	RETURN
	END
