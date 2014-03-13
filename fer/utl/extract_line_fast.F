	SUBROUTINE EXTRACT_LINE_FAST( axis, block, bad_flag, i, j, l, m, n,
     .		cxloz, cxhiz,
     .		slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif,
     .		zline, klo, khi )


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
* Extract a single line of data from a memory variable grid.
* Identify contiguously valid data region and report in klo, khi

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V690:  12/13 *sh* derived from EXTRACT_Z_LINE

* calling argument declarations:
	INTEGER axis,
     .		slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		sloe, shie, slof, shif, i, j, l, m, n, 
     .		klo, khi, cxloz, cxhiz
	REAL    bad_flag,
     .		block( slox:shix,sloy:shiy,sloz:shiz,
     .		       slot:shit,sloe:shie,slof:shif ),
     .		zline(cxloz:cxhiz)

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
 100	zline(k) = block(i,j,k,l,m,n)

* find the start and end of valid data
	DO 200 k = cxloz, cxhiz

* ... skip invalid values at start of line
	   IF ( searching_for_valid ) THEN
	      IF (zline(k) .EQ. bad_flag) GOTO 200
	      searching_for_valid = .FALSE.
	      klo = k
* ... terminate if invalid point encountered after valid points
	   ELSEIF ( zline(k) .EQ. bad_flag ) THEN
	      khi = k-1
	      GOTO 300
	   ENDIF

 200	CONTINUE
	khi = cxhiz

* successful completion
 300	RETURN
	END
