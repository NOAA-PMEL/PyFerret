	SUBROUTINE TM_CHECK_LINE ( line, npts, reversed, vname, vlen, 
     .      is_double, its_irregular, use_strict, regular, misordered)

*
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
* Check coordinate data from a netCDF file
*
* Check for misordered or repeated data.
* If repeated values, apply micro-adjusting, or report as a warning if the 
* user requested /STRICT.

* Check if axis is actually regularly spaced (though not flagged as "even")
* Use the precision of the data  in the file.

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG

* revision 0.0 - 6/17
* V72 6/17 *acm* for trac enhancement #767 -- dynamic coordinate storage 


        include 'tmap_errors.parm'
        include 'xio.cmn_text'
        external xio_data

* argument declarations

	LOGICAL reversed, is_double, has_repeated, its_irregular, 
     .       use_strict, regular, misordered
	REAL line(*)
	INTEGER npts, vlen
	CHARACTER*(*) vname


* local declarations
	LOGICAL TM_DFPEQ_TOL, TM_FPEQ_EPS
	REAL tmp8, micro_adj, epsilon_21, depsilon_42, epsilon, depsilon,
     .       dfirst_delta, ddelta, delta, firs_coord, last_coord, first_delta
	INTEGER i, istat
	CHARACTER*12 TM_LEFINT

	misordered = .FALSE.  ! valid until proven otherwise

* (From Wikipedia on precision:)
* Single precision, called "float" in the C language family, and "real" 
* or "real*4" in Fortran. This is a binary format that occupies 32 bits (4 bytes) 
* and its significand has a precision of 24 bits (about 7 decimal digits).
* Double precision, called "double" in the C language family, and "double precision" 
* or "real*8" in Fortran. This is a binary format that occupies 64 bits (8 bytes) 
* and its significand has a precision of 53 bits (about 16 decimal digits).

      epsilon_21 =  2.**(-21)
      depsilon_42 =  2.**(-42) 

* if the coordinates are decreasing-ordered then reverse them

	IF (npts .GT. 1) THEN
	   first_delta = line(2) - line(1)
	   reversed = first_delta .LT. 0.0
	ENDIF

	IF (reversed) THEN
	   DO i = 1, npts/2
	      tmp8 = line(i)
	      line(i) = line(npts+1-i)
	      line(npts+1-i) = tmp8
	   ENDDO
	ENDIF

* If repeated values, apply micro-adjusting. 
* Or report as a warning if the user requested /STRICT.

         misordered = .FALSE.
         DO i = 2, npts
           IF (line(i) .LT. line(i-1) ) 
     .	      misordered = .TRUE.
	   IF (use_strict .AND. (line(i) .LE. line(i-1)) )
     .	      misordered = .TRUE.

	   IF (misordered) EXIT ! from loop
         ENDDO

* 0726348357

* out of order axis coordinates.
* If reversed above, remove that designation to 
* just hand the user the original data

         IF (misordered) THEN
            CALL TM_NOTE(
     .          'Coordinates out of order or missing on axis '//
     .           vname(:vlen)//' at subscript '//TM_LEFINT(i, istat), 
     .           lunit_errors)
            CALL TM_NOTE(
     .          'A dummy axis of subscripts will be used', lunit_errors)
     
            IF (reversed) THEN
               DO i = 1, npts/2
	         tmp8 = line(i)
	         line(i) = line(npts+1-i)
	         line(npts+1-i) = tmp8
               ENDDO
            ENDIF

            GOTO 1000
         ENDIF

         misordered = .FALSE.
         micro_adj = 1E-7 *
     .		(line(npts) - line(1))

* ... are there repeated values?

	 CALL TM_CHECK_COORDS (line, npts, is_double, use_strict, 
     .                         has_repeated, misordered, micro_adj, 
     .                         epsilon, depsilon)

	 IF (misordered) GOTO 2000
         IF ( has_repeated .AND. .NOT.misordered ) THEN
            CALL TM_NOTE('Axis has repeated values -- micro-adjusting '//
     .           vname(:vlen), lunit_errors)
         ENDIF

* Check if axis is actually regularly spaced (though not flagged as "even")

        IF (its_irregular) GOTO 1000  ! If the file says its irreg, treat it at such

* See if double-precision coordinates are regularly-spaced
	epsilon = 0.
	depsilon = 0.

        IF (is_double) THEN
           dfirst_delta = line(2) - line(1)
           depsilon = depsilon_42 * 2.*(ABS(line(npts)) / first_delta )
           depsilon = depsilon_42 * 2.*(ABS(line(1)) / first_delta )

           DO i = 2, npts
              ddelta = line(i) - line(i-1)
              IF (.NOT. TM_DFPEQ_TOL(dfirst_delta, ddelta, depsilon) ) GOTO 1000
           ENDDO

* See if single-precision (or integer??) coordinates are regularly-spaced
	ELSE
           last_coord = line(npts)
           firs_coord = line(1)
           first_delta = line(2) - line(1)
           epsilon = epsilon_21 * 2.*(ABS(line(npts)) / first_delta )
           epsilon = epsilon_21 * 2.*(ABS(line(1)) / first_delta )

           DO  i = 3, npts
              delta = line(i) - line(i-1)
              IF (.NOT. TM_FPEQ_EPS(epsilon, first_delta, delta) ) GOTO 1000
           ENDDO
	ENDIF
	regular = .TRUE.
	RETURN

 1000	CONTINUE
        regular = .FALSE.

 2000	RETURN
	END
