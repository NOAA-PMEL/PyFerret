      SUBROUTINE OPERATE (	operation, num_com, com_cx,
     .				com1, mr1,
     .				com2, mr2,
     .				com3, mr3,
     .				com4, mr4,
     .				res, mres )

* perform an operation involving portions of 0,1,2 or 3 input grids
* and producing an entire result grid

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/20/89-8/3/89 - based on GRID_OPERATION
*		 - 4-D symmetric where components may have differing dims
*		 - eliminated derivatives (now "@" transformations)
*		 - added IGNORE0 and EXP
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
*	10/26/89 - corrected bad flag bug by introducing bad_res
*	11/28/89 - reordering of functions for change to in-fix notation
*	 3/19/90 - added ATAN,ATAN2,ASIN,ACOS
*	 3/29/90 - added RANDU,RANDN
* Unix/RISC port - 10/91 - corrected improper calculation of MISSING func
*                          for cases where arg1 was ok but arg2 was bad
* V230:  8/5/92 - added new functions RHO_UN and THETA_UN
*               - upped to 4 the number of function arguments allowed
*       8/19/92 - LOG and LN args must be positive
*                 EXP arg is limited to (approx.) 77
*                 RHO salinity arg must be non-negative
*	6/93	- correct subscript bug introduced 8/19/92
* V320: 11/94   - added new function DAY1900(year, month, day)
* Linux Port  12/96 *kob* - had to pass 0.0 to RANDU and RANDN rather 
*			    than the 0 that was being passed. F90 
*			    complained about inconsistent datatypes
#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	operation, num_com, com_cx(4), mr1, mr2, mr3, mr4, mres
	REAL    com1( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		com2( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		com3( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .		com4( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit ),
     .		 res( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit )

* internal variable declarations:
	LOGICAL	first
	INTEGER CX_DIM_LEN, i, j, k, l, icom, idim,
     .		i1, j1, k1, l1, i2, j2, k2, l2,
     .		i3, j3, k3, l3, i4, j4, k4, l4,
     .		di1, dj1, dk1, dl1, di2, dj2, dk2, dl2,
     .		di3, dj3, dk3, dl3, di4, dj4, dk4, dl4,
     .		si1, sj1, sk1, sl1, si2, sj2, sk2, sl2,
     .		si3, sj3, sk3, sl3, si4, sj4, sk4, sl4
	REAL	RANDU, RANDN, RHO_UNESCO, THETA_FOFF,
     .		DAYS_FROM_DAY0,
     .          bad1, bad2, bad3, bad4, bad_res

* equivalence conveniences
* "inc" is the increments for each axis of each component
* "lo"  is the starting subscript (minus 1) for each axis of each component
	INTEGER	inc(4,4), lo(4,4)
	EQUIVALENCE
     .	(lo(1,1),si1), (lo(2,1),sj1), (lo(3,1),sk1), (lo(4,1),sl1),
     .	(lo(1,2),si2), (lo(2,2),sj2), (lo(3,2),sk2), (lo(4,2),sl2),
     .	(lo(1,3),si3), (lo(2,3),sj3), (lo(3,3),sk3), (lo(4,3),sl3),
     .	(lo(1,4),si4), (lo(2,4),sj4), (lo(3,4),sk4), (lo(4,4),sl4),
     .	(inc(1,1),di1),(inc(2,1),dj1),(inc(3,1),dk1),(inc(4,1),dl1),
     .	(inc(1,2),di2),(inc(2,2),dj2),(inc(3,2),dk2),(inc(4,2),dl2),
     .	(inc(1,3),di3),(inc(2,3),dj3),(inc(3,3),dk3),(inc(4,3),dl3),
     .	(inc(1,4),di4),(inc(2,4),dj4),(inc(3,4),dk4),(inc(4,4),dl4)

* internal parameter declarations:
	REAL*8          pdays_by_1900
	PARAMETER     ( pdays_by_1900 = 59958230400.0 / (60.*60.*24.) )

* flag(s) for bad or missing values
	bad1 = mr_bad_data ( mr1 )
	bad2 = mr_bad_data ( mr2 )
	bad3 = mr_bad_data ( mr3 )
	bad4 = mr_bad_data ( mr4 )
	bad_res = mr_bad_data( mres )

* compute the delta increment for each axis of each component
	DO 10 icom = 1, num_com
	DO 10 idim = 1, 4
	   IF ( CX_DIM_LEN(idim,com_cx(icom)) .EQ. 1 ) THEN
	      inc(idim,icom) = 0
	   ELSE
	      inc(idim,icom) = 1
	   ENDIF
 10	CONTINUE

* compute the starting subscript for each axis of each component
* (pre-decrement by 1 delta for looping ahead)
	DO 20 icom = 1, num_com
	DO 20 idim = 1, 4
 20	lo(idim,icom) = cx_lo_ss(com_cx(icom),idim) - inc(idim,icom)

* operators, functions, logic structures
	GOTO (   100, 200, 300, 400, 500, 600, 700, 800, 900,1000,
     .		1100,1200,1300,1400,1500,1600,1700,1800,
     .
     .		1900,2000,
     .		2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,
     .		3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,
     .		4100,4200,4300,
     .
     .		4400,4500,4600					   ) operation

* ----------------------------------------------------------------------------
* ------- OPERATORS -------------
*
* "+"
 100	l1 = sl1
	l2 = sl2
	DO 110 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 110 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 110 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 110 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = com1(i1,j1,k1,l1) + com2(i2,j2,k2,l2)
	   ENDIF

 110	CONTINUE
	RETURN

* "-"
 200	l1 = sl1
	l2 = sl2
	DO 210 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 210 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 210 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 210 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = com1(i1,j1,k1,l1) - com2(i2,j2,k2,l2)
	   ENDIF

 210	CONTINUE
	RETURN
* "*"
 300	l1 = sl1
	l2 = sl2
	DO 310 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 310 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 310 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 310 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = com1(i1,j1,k1,l1) * com2(i2,j2,k2,l2)
	   ENDIF

 310	CONTINUE
	RETURN

* "/"
 400	l1 = sl1
	l2 = sl2
	DO 410 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 410 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 410 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 410 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2
     .	   .OR. com2(i2,j2,k2,l2) .EQ. 0.0   ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = com1(i1,j1,k1,l1) / com2(i2,j2,k2,l2)
	   ENDIF

 410	CONTINUE
	RETURN

* "^"
 500	l1 = sl1
	l2 = sl2
	DO 510 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 510 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 510 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 510 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2
     .	   .OR.    ( com1(i1,j1,k1,l1) .EQ. 0.0
     .	       .AND. com2(i2,j2,k2,l2) .LT. 0.0 ) ) THEN
	      res(i,j,k,l) = bad_res
	   ELSEIF ( com2(i2,j2,k2,l2).EQ.FLOAT(INT(com2(i2,j2,k2,l2))) ) THEN
	      res(i,j,k,l) = com1(i1,j1,k1,l1) ** INT(com2(i2,j2,k2,l2))
	   ELSEIF ( com1(i1,j1,k1,l1) .LT. 0.0 )	THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = com1(i1,j1,k1,l1) ** com2(i2,j2,k2,l2)
	   ENDIF

 510	CONTINUE
	RETURN

* "AND" - logical AND of 2 masks (any non-zero data regarded as .TRUE.)
 600	l1 = sl1
	l2 = sl2
	DO 610 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 610 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 610 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 610 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1).NE.0.0
     .	     .AND. com2(i2,j2,k2,l2).NE.0.0 ) THEN
		 res(i,j,k,l) = 1.0
	      ELSE
		 res(i,j,k,l) = 0.0
	      ENDIF
	   ENDIF
 610	CONTINUE
	RETURN

* "OR" - logical OR of 2 masks (any non-zero data regarded as .TRUE.)
 700	l1 = sl1
	l2 = sl2
	DO 710 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 710 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 710 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 710 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1).NE.0.0
     .	      .OR. com2(i2,j2,k2,l2).NE.0.0 ) THEN
		 res(i,j,k,l) = 1.0
	      ELSE
		 res(i,j,k,l) = 0.0
	      ENDIF
	   ENDIF
 710	CONTINUE
	RETURN

* "GT" - mask of 1's where grid 1 exceeds grid 2, 0's elsewhere
 800	l1 = sl1
	l2 = sl2
	DO 810 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 810 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 810 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 810 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1) .GT. com2(i2,j2,k2,l2) ) THEN
		 res(i,j,k,l) = 1.0
	      ELSE
		 res(i,j,k,l) = 0.0
	      ENDIF
	   ENDIF
 810	CONTINUE
	RETURN

* "GE" - mask of 1's where grid 1 exceeds or equals grid 2, 0's elsewhere
 900	l1 = sl1
	l2 = sl2
	DO 910 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 910 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 910 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 910 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1) .GE. com2(i2,j2,k2,l2) ) THEN
		 res(i,j,k,l) = 1.0
	      ELSE
		 res(i,j,k,l) = 0.0
	      ENDIF
	   ENDIF
 910	CONTINUE
	RETURN

* "LT" - mask of 1's where grid 1 is less than grid 2, 0's elsewhere
 1000	l1 = sl1
	l2 = sl2
	DO 1010 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 1010 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 1010 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 1010 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1) .LT. com2(i2,j2,k2,l2) ) THEN
		 res(i,j,k,l) = 1.0
	      ELSE
		 res(i,j,k,l) = 0.0
	      ENDIF
	   ENDIF
 1010	CONTINUE
	RETURN

* "LE" - mask of 1's where grid 1 is less than or equal to grid 2, 0's elsewhere
 1100	l1 = sl1
	l2 = sl2
	DO 1110 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 1110 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 1110 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 1110 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1) .LE. com2(i2,j2,k2,l2) ) THEN
		 res(i,j,k,l) = 1.0
	      ELSE
		 res(i,j,k,l) = 0.0
	      ENDIF
	   ENDIF
 1110	CONTINUE
	RETURN

* "EQ" - mask of 1's where grids are equal, 0's elsewhere
 1200	l1 = sl1
	l2 = sl2
	DO 1210 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 1210 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 1210 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 1210 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1) .EQ. com2(i2,j2,k2,l2) ) THEN
		 res(i,j,k,l) = 1.0
	      ELSE
		 res(i,j,k,l) = 0.0
	      ENDIF
	   ENDIF
 1210	CONTINUE
	RETURN

* "NE" - mask of 1's where grids are not equal, 0's elsewhere
 1300	l1 = sl1
	l2 = sl2
	DO 1310 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 1310 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 1310 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 1310 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1) .NE. com2(i2,j2,k2,l2) ) THEN
		 res(i,j,k,l) = 1.0
	      ELSE
		 res(i,j,k,l) = 0.0
	      ENDIF
	   ENDIF
 1310	CONTINUE
	RETURN

 1400	CONTINUE
 1500	CONTINUE
 1600	CONTINUE
 1700	CONTINUE
 1800	CONTINUE
	STOP 'OPERATOR NOT IMPLEMENTED'
* -----------------------------------------------------------------------------
* -------- FUNCTIONS ------------

* "EXP" - exponential function
 1900	DO 1910 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 1910 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 1910 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 1910 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1
     .     .OR. com1(i,j,k,l) .GT. 77.0 ) THEN    ! 8/92
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = EXP(com1(i,j,k,l))
	   ENDIF
 1910	CONTINUE
	RETURN

* "LOG"
 2000	DO 2010 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 2010 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 2010 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 2010 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1
     .    .OR.  com1(i,j,k,l) .LE. 0.0 ) THEN    ! 8/92
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = LOG10( com1(i,j,k,l) )
	   ENDIF
 2010	CONTINUE
	RETURN

* "MAX"
 2100	l1 = sl1
	l2 = sl2
	DO 2110 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 2110 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 2110 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 2110 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = MAX( com1(i1,j1,k1,l1), com2(i2,j2,k2,l2) )
	   ENDIF

 2110	CONTINUE
	RETURN

* "MIN"
 2200	l1 = sl1
	l2 = sl2
	DO 2210 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 2210 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 2210 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 2210 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = MIN( com1(i1,j1,k1,l1), com2(i2,j2,k2,l2) )
	   ENDIF
 2210	CONTINUE
	RETURN

* "INT"
 2300	DO 2310 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 2310 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 2310 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 2310 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = INT( com1(i,j,k,l) )
	   ENDIF
 2310	CONTINUE
	RETURN

* "ABS"
 2400	DO 2410 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 2410 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 2410 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 2410 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = ABS( com1(i,j,k,l) )
	   ENDIF
 2410	CONTINUE
	RETURN

* "SIN"
 2500	DO 2510 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 2510 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 2510 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 2510 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = SIN( com1(i,j,k,l) )
	   ENDIF
 2510	CONTINUE
	RETURN

* "COS"
 2600	DO 2610 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 2610 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 2610 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 2610 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = COS( com1(i,j,k,l) )
	   ENDIF
 2610	CONTINUE
	RETURN

* "TAN"
 2700	DO 2710 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 2710 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 2710 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 2710 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = TAN( com1(i,j,k,l) )
	   ENDIF
 2710	CONTINUE
	RETURN

* "LN"
 2800	DO 2810 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 2810 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 2810 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 2810 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1
     .    .OR.  com1(i,j,k,l) .LE. 0.0 ) THEN    ! 8/92
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = LOG( com1(i,j,k,l) )
	   ENDIF
 2810	CONTINUE
	RETURN

* "MOD"
 2900	l1 = sl1
	l2 = sl2
	DO 2910 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 2910 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 2910 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 2910 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = MOD( com1(i1,j1,k1,l1), com2(i2,j2,k2,l2) )
	   ENDIF
 2910	CONTINUE
	RETURN

* "MISSING" - replace missing value flag with values in grid 2
 3000	l1 = sl1
	l2 = sl2
	DO 3010 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 3010 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 3010 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 3010 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .NE. bad1 ) THEN
              res(i,j,k,l) = com1(i1,j1,k1,l1)
           ELSEIF ( com2(i2,j2,k2,l2) .NE. bad2 ) THEN
              res(i,j,k,l) = com2(i2,j2,k2,l2)
           ELSE
              res(i,j,k,l) = bad_res
           ENDIF

 3010	CONTINUE
	RETURN

* "IGNORE0" - replace zeros with bad flags
 3100	DO 3110 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 3110 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 3110 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 3110 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSEIF ( com1(i,j,k,l) .EQ. 0 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = com1(i,j,k,l)
	   ENDIF
 3110	CONTINUE
	RETURN

* "ATAN"
 3200	DO 3210 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 3210 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 3210 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 3210 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = ATAN( com1(i,j,k,l) )
	   ENDIF
 3210	CONTINUE
	RETURN

* "ATAN2" - 2 argument arc tangent
 3300	l1 = sl1
	l2 = sl2
	DO 3310 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 3310 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 3310 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 3310 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2) .EQ. bad2 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSEIF ( com1(i1,j1,k1,l1) .EQ. 0.0
     .	   .AND.    com2(i2,j2,k2,l2) .EQ. 0.0 ) THEN
	      res(i,j,k,l) = bad_res	! indeterminant
	   ELSE
	      res(i,j,k,l) = ATAN2( com1(i1,j1,k1,l1),com2(i2,j2,k2,l2) )
	   ENDIF

 3310	CONTINUE
	RETURN

* "ASIN"
 3400	DO 3410 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 3410 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 3410 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 3410 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1
     .     .OR. com1(i,j,k,l) .LT. -1.
     .     .OR. com1(i,j,k,l) .GT. +1. ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = ASIN( com1(i,j,k,l) )
	   ENDIF
 3410	CONTINUE
	RETURN

* "ACOS"
 3500	DO 3510 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 3510 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 3510 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 3510 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1
     .     .OR. com1(i,j,k,l) .LT. -1.
     .     .OR. com1(i,j,k,l) .GT. +1. ) THEN
	      res(i,j,k,l) = bad_res
	   ELSE
	      res(i,j,k,l) = ACOS( com1(i,j,k,l) )
	   ENDIF
 3510	CONTINUE
	RETURN

* "RANDU" - random uniform
* use the first value as the seed
 3600	first = .TRUE.
	DO 3610 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 3610 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 3610 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 3610 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSEIF ( first ) THEN
* ... note: floating point representation is passed as seed integer
	      res(i,j,k,l) = RANDU( com1(i,j,k,l) )
	      first = .FALSE.
	   ELSE
	      res(i,j,k,l) = RANDU(0.0)			!kob 12/96
	   ENDIF
 3610	CONTINUE
	RETURN

* "RANDN" - random normal
* use the first value as the seed
 3700	first = .TRUE.
	DO 3710 l = mr_lo_s4(mres), mr_hi_s4(mres)
	 DO 3710 k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO 3710 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 3710 i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l) .EQ. bad1 ) THEN
	      res(i,j,k,l) = bad_res
	   ELSEIF ( first ) THEN
* ... note: floating point representation is passed as seed integer
	      res(i,j,k,l) = RANDN( com1(i,j,k,l) )
	      first = .FALSE.
	   ELSE
	      res(i,j,k,l) = RANDN(0.0)			!kob 12/96
	   ENDIF
 3710	CONTINUE
	RETURN

* "RHO" - UNESCO equation of state:
* rho = rho(salinity, temperature, pressure)
* three components - each may be a different XYZT shape
 3800	l1 = sl1
	l2 = sl2
	l3 = sl3
	DO 3810 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	l3 = l3 + dl3
	k1 = sk1
	k2 = sk2
	k3 = sk3
	 DO 3810 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 k3 = k3 + dk3
	 j1 = sj1
	 j2 = sj2
	 j3 = sj3
	  DO 3810 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  j3 = j3 + dj3
	  i1 = si1
	  i2 = si2
	  i3 = si3
	   DO 3810 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2
	   i3 = i3 + di3

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .     .OR. com2(i2,j2,k2,l2) .EQ. bad2
     .     .OR. com3(i3,j3,k3,l3) .EQ. bad3
     .     .OR. com1(i1,j1,k1,l1) .LT. 0.0   ) THEN  ! salt>0 8/92
              res(i,j,k,l) = bad_res
           ELSE
              res(i,j,k,l) = RHO_UNESCO( com1(i1,j1,k1,l1),
     .                                   com2(i2,j2,k2,l2),
     .                                   com3(i3,j3,k3,l3) )
           ENDIF

 3810	CONTINUE
	RETURN

* "THETA" - potential temperature from BRYDEN,H.,1973,DEEP-SEA RES
*           and FOFONOFF,N,M,1977,DEEP-SEA RES
* theta = theta(salinity, temperature, pressure,reference_pressure)
* three components - each may be a different XYZT shape
 3900	l1 = sl1
	l2 = sl2
	l3 = sl3
	l4 = sl4
	DO 3910 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	l3 = l3 + dl3
	l4 = l4 + dl4
	k1 = sk1
	k2 = sk2
	k3 = sk3
	k4 = sk4
	 DO 3910 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 k3 = k3 + dk3
	 k4 = k4 + dk4
	 j1 = sj1
	 j2 = sj2
	 j3 = sj3
	 j4 = sj4
	  DO 3910 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  j3 = j3 + dj3
	  j4 = j4 + dj4
	  i1 = si1
	  i2 = si2
	  i3 = si3
	  i4 = si4
	   DO 3910 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2
	   i3 = i3 + di3
	   i4 = i4 + di4

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .     .OR. com2(i2,j2,k2,l2) .EQ. bad2
     .     .OR. com3(i3,j3,k3,l3) .EQ. bad3
     .     .OR. com4(i4,j4,k4,l4) .EQ. bad4 ) THEN
              res(i,j,k,l) = bad_res
           ELSE
              res(i,j,k,l) = THETA_FOFF( com1(i1,j1,k1,l1),
     .                                   com2(i2,j2,k2,l2),
     .                                   com3(i3,j3,k3,l3),
     .                                   com4(i4,j4,k4,l4) )
           ENDIF

 3910	CONTINUE
	RETURN

* "DAYS1900(year, month, day)" - number of days since 1900
* three components - each may be a different XYZT shape
 4000	l1 = sl1
	l2 = sl2
	l3 = sl3
	DO 4010 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	l3 = l3 + dl3
	k1 = sk1
	k2 = sk2
	k3 = sk3
	 DO 4010 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 k3 = k3 + dk3
	 j1 = sj1
	 j2 = sj2
	 j3 = sj3
	  DO 4010 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  j3 = j3 + dj3
	  i1 = si1
	  i2 = si2
	  i3 = si3
	   DO 4010 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2
	   i3 = i3 + di3

	   IF ( com1(i1,j1,k1,l1) .EQ. bad1
     .     .OR. com2(i2,j2,k2,l2) .EQ. bad2
     .     .OR. com3(i3,j3,k3,l3) .EQ. bad3
     .     .OR. com1(i1,j1,k1,l1) .LT. 1900.0
     .     .OR. com2(i1,j1,k1,l1) .LT. 1.0
     .     .OR. com3(i1,j1,k1,l1) .LT. 0.0    ) THEN
              res(i,j,k,l) = bad_res
           ELSE
              res(i,j,k,l) = DAYS_FROM_DAY0( pdays_by_1900,
     .					     INT(com1(i1,j1,k1,l1)),
     .                                       INT(com2(i2,j2,k2,l2)),
     .                                       INT(com3(i3,j3,k3,l3)) )
           ENDIF

 4010	CONTINUE
	RETURN

 4100	CONTINUE
 4200	CONTINUE
 4300	CONTINUE
	STOP 'FUNCTION NOT IMPLEMENTED'

* ----------------------------------------------------------------------------
* ------- LOGIC STRUCTURES -------------
*
 4400	CONTINUE
* "THEN" - comes from IF (condition) THEN (choice)   ... no "ELSE" given
* component 1 is the condition.  component 2 is the choice
 4500	l1 = sl1
	l2 = sl2
	DO 4510 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 4510 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 4510 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 4510 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2
	   IF ( com1(i1,j1,k1,l1) .NE. bad1
     .	  .AND. com1(i1,j1,k1,l1) .NE. 0.0
     .	  .AND. com2(i2,j2,k2,l2) .NE. bad2	 ) THEN
	      res(i,j,k,l) = com2(i2,j2,k2,l2)
	   ELSE
	      res(i,j,k,l) = bad_res
	   ENDIF

 4510	CONTINUE
	RETURN
* "ELSE" - comes from IF (condition) THEN (choice1) ELSE (choice2)
* component 1 is the condition.  components 2 and 3 are the choices
 4600	l1 = sl1
	l2 = sl2
	l3 = sl3
	DO 4610 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	l3 = l3 + dl3
	k1 = sk1
	k2 = sk2
	k3 = sk3
	 DO 4610 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 k3 = k3 + dk3
	 j1 = sj1
	 j2 = sj2
	 j3 = sj3
	  DO 4610 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  j3 = j3 + dj3
	  i1 = si1
	  i2 = si2
	  i3 = si3
	   DO 4610 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2
	   i3 = i3 + di3
	   IF ( com1(i1,j1,k1,l1) .NE. bad1
     .	  .AND. com1(i1,j1,k1,l1) .NE. 0.0 ) THEN
	      IF ( com2(i2,j2,k2,l2) .NE. bad2 ) THEN
	         res(i,j,k,l) = com2(i2,j2,k2,l2)
	      ELSE
	         res(i,j,k,l) = bad_res
	      ENDIF
	   ELSE
	      IF ( com3(i3,j3,k3,l3) .NE. bad3 ) THEN
	         res(i,j,k,l) = com3(i3,j3,k3,l3)
	      ELSE
	         res(i,j,k,l) = bad_res
	      ENDIF
	   ENDIF
 4610	CONTINUE
	RETURN

* error exits
! 9000	RETURN
	END
