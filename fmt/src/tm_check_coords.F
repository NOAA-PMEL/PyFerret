	SUBROUTINE TM_CHECK_COORDS (line, npts, is_double, use_strict, 
     .                              has_repeated, misordered, micro, 
     .                              epsilon, depsilon)

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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
* Ferret v7.1 *acm* 1/13/2017
* V72 6/17 *acm* for trac enhancement #767 -- dynamic coordinate storage

* This routine replaces identical logic in cd_get_1_axis and xeq_define, 
* checking for repeated coordinates and doing micro-adjusting if possible.
*
* Check if coordinates in coordinate array have repeated coordintes as in cd_get_1_axis 
* or xeq_define.Apply micro-adjusging to correct this, unless flag use_strict is set.

* Arguments
* line  coordinates for the line
* is_double, for netCDF files, flag is true if the coordinates in the file are double precision.
* use_strict, for netCDF files, flag is true if the user requested /STRICT; do not do micro-adjusting
* micro      If positive, use this value for the micro-adjustment, otherwise compute it.

* Returns:
*  has_repeated		True if coordinates found to be repeating.
*  misordered		True if coordinates cannot be fixed
*  epsilon		For single-precision input coordinates, the epsilon used to compare coordinates
*  depsilon		For double-precision input coordinates,    ""            ""


        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'xio.cmn_text'
        external xio_data

* argument declarations

	LOGICAL is_double, has_repeated, misordered, use_strict
	INTEGER	npts
	REAL	line(*), micro, epsilon, depsilon

* Local variable declarations
	LOGICAL TM_DFPEQ_TOL, TM_FPEQ_EPS
	REAL	dfirst_delta, first_delta, micro_adj, madj, delta, zero
	REAL epsilon_23, depsilon_46
	INTEGER i, i0, i1, i2, i3, vlen

* Initialize
	epsilon_23 =  2.**(-23) 
	depsilon_46 =  2.**(-46) 
	
	has_repeated = .FALSE.
	misordered = .FALSE.

* Set epsilon value for comparing coordinates

	epsilon = 0.
	depsilon = 0.

	IF (npts .LT. 2) RETURN

        IF (is_double) THEN

           dfirst_delta = line(2) - line(1)
	   IF (dfirst_delta .LT. depsilon_46) THEN
	      DO i = 2, npts
                dfirst_delta = line(i) - line(i-1) 
		IF (dfirst_delta .GE. depsilon_46) EXIT  ! from loop
	      ENDDO
	      IF (dfirst_delta .LT. depsilon_46) THEN
	         misordered = .TRUE.
		 GOTO 1000
	      ENDIF
	   ENDIF
           depsilon = depsilon_46 * 2.*(ABS(line(npts)) / dfirst_delta )
           depsilon = depsilon_46 * 2.*(ABS(line(1)) / dfirst_delta )

        ELSE

           dfirst_delta = line(2) - line(1)
           epsilon = epsilon_23 * 2.*(ABS(line(npts)) / first_delta )
           epsilon = epsilon_23 * 2.*(ABS(line(1)) / first_delta )
	   
	   IF (first_delta .LT. epsilon_23) THEN
	      DO i = 1, npts
                first_delta = line(i+1) - line(i) 
		IF (first_delta .GE. epsilon_23) EXIT  ! from loop
	      ENDDO
	      IF (first_delta .LT. epsilon_23) THEN
	         misordered = .TRUE.
		 GOTO 1000
	      ENDIF
	   ENDIF

	ENDIF

* If repeated values, apply micro-adjusting. 
* Or report as a warning if the user requested /STRICT.

	IF (micro .GT. 0.) THEN
	   micro_adj = micro
	ELSE
	   micro_adj = 1E-7 * (line(npts) - line(1))
	ENDIF

* ... are there repeated values?
	 
	IF (is_double) THEN
	   zero = 0.
	   DO 110 i1 = 2, npts
              delta = line(i1) - line(i1-1)
              IF (TM_DFPEQ_TOL(delta, zero, depsilon) ) THEN
	          has_repeated = .TRUE.
		  GOTO 130
	      ENDIF
 110       CONTINUE

	ELSE
           zero = 0.0
	   DO 120 i1 = 2, npts
	      delta = line(i1) - line(i1-1)
	      IF (TM_FPEQ_EPS(epsilon, delta, zero) ) THEN
	          has_repeated = .TRUE.
		  GOTO 130
	      ENDIF
 120       CONTINUE

	ENDIF

 130	CONTINUE

* We are done if repeated values not found, or if the user asked for /STRICT

	IF (.NOT. has_repeated .OR. use_strict) GOTO 1000

* ... make micro-adjustments if axis has repeated values (4/99)

        IF (micro_adj .EQ. 0.) THEN
           i1 = 1
           misordered = .TRUE.
           GOTO 1000
        ENDIF

        i0 = 1

* --- adjust coordinates coming in as double-precision

        IF (is_double) THEN

           DO 170  i1 = 2, npts
	    
               delta = line(i1) - line(i1-1)
               IF (TM_DFPEQ_TOL(delta, zero, depsilon) ) THEN
*    ... locate end of repeating block
                  DO 140 i2 = i1, npts
                        delta = line(i2) - line(i1)
                        IF ( delta .GT. depsilon ) GOTO 150
 140              CONTINUE
                  i2 = npts + 1    ! one beyond last value
                  i0 = i1 - 2
                  i3 = i2 - 1
*    ... micro-adjust repeating values from i1 to i2-1
 150              CONTINUE

                  delta = line(i2)-line(i1)
                  IF (.NOT.TM_DFPEQ_TOL(delta, zero, depsilon) .AND. 
     .                 line(i2).GT.line(i1)) THEN
                     i0 = i1-1
                     i3 = i2
                  ENDIF

* Check and readjust the size of micro_adj if too small
                  madj = micro_adj
                  IF (micro_adj*(i2-i1+1) .GT. 
     .                line(i3) - line(i0) ) THEN
                     madj = (line(i3)-line(i0))/(i2-i1+1)
                     IF (madj .EQ. 0.)  THEN
	                misordered = .TRUE.
	                GOTO 1000
	             ENDIF
	          ENDIF

                  DO 160 i3 = i1, i2-1
                     line(i3) = line(i3) + (i3-i1+1)* madj
 160              CONTINUE
                  
                  delta = line(i2)-line(i2-1)
                  IF ( i2 .LT. npts .AND. .NOT.TM_DFPEQ_TOL(delta, zero, depsilon) .AND.
     .		      (line(i2-1) .GE. line(i2)))  THEN
	             misordered = .TRUE.
	             GOTO 1000
                  ENDIF
              ENDIF
 170       CONTINUE


* --- adjust coordinates coming in as single-precision
	ELSE ! single

           DO 270  i1 = 2, npts
	    
              delta = line(i1) - line(i1-1)
              IF (TM_FPEQ_EPS(epsilon, delta, zero) ) THEN
*    ... locate end of repeating block
                 DO 240 i2 = i1, npts
                    delta = line(i2) - line(i1)
                    IF (delta .GT. epsilon) GOTO 250
 240             CONTINUE
                 i2 = npts + 1    ! one beyond last value
                 i0 = i1 - 2
                 i3 = i2 - 1
*    ... micro-adjust repeating values from i1 to i2-1
 250             CONTINUE

                 delta = line(i2)-line(i1)
                 IF (.NOT.TM_FPEQ_EPS(epsilon, delta, zero) .AND. 
     .               line(i2).GT.line(i1)) THEN
                    i0 = i1-1
                    i3 = i2
                 ENDIF

* Check and readjust the size of micro_adj if too small
                 madj = micro_adj
                 IF (micro_adj*(i2-i1+1) .GT. 
     .               line(i3) - line(i0) ) THEN
                    madj = (line(i3)-line(i0))/(i2-i1+1)
                    IF (madj .EQ. 0.)  THEN
	                   misordered = .TRUE.
	                   GOTO 1000
                    ENDIF
                  ENDIF
                  DO 260 i3 = i1, i2-1
                     line(i3) = line(i3) + (i3-i1+1)* madj
 260              CONTINUE
                  
		  
                  delta = line(i2)-line(i2-1)
                  IF ( i2 .LT. npts .AND. .NOT.TM_FPEQ_EPS(epsilon, delta, zero) .AND.
     .		      (line(i2-1) .GE. line(i2)))  THEN
                     misordered = .TRUE.
	             GOTO 1000
                 ENDIF
              ENDIF  
 270       CONTINUE

	ENDIF ! single or double

 1000	CONTINUE
	RETURN

	END
