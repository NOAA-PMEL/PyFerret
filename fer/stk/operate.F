      SUBROUTINE OPERATE (	operation, num_com, ifv, com_cx,
     .				com1, mr1,
     .				com2, mr2,
     .				com3, mr3,
     .				com4, mr4,
     .				res, mres )

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
*
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
* v491: 4/98 *kob* - had to fix incorrect array references using i1,j1,k1 for
*                    arrays com2, com3
* V510 *sh* 1/00 - fixed argument checks on DAYS1900
*                - allow pre-1900, forbid month>12
*      5/00 *sh* - bug fix in line checking month>12
* v5.40*acm* add comment: days1900 returns days since 1-jan-1900, STANDARD CALENDAR
* V600 4/06  *acm*  return the result from days_from_day0 in dummy, for 64-bit build
* V671 7/11  *acm* Ticket 1868 IFV - masking expressions with ifValid, treating 
*		   zero as valid.
* V672 9/11  *acm* add RANDU2,RANDN2.  The way RANDN works, with SAVEd variables,
*                  it's not going to work well as an external function.
* V6.8 for double-precision Ferret do the RANDU calculation
*      with a real*4 argument. Gives consistent results with single-p.
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V7.43 Issue 1891 Compute v^2 as v*v for efficiency
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*  1/2023 *acm* Code cleanup: Replace the computed GOTO statement 
*                  "GOTO ( 100, 200, 300..) op"  with IF blocks

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	LOGICAL ifv
	INTEGER	operation, num_com, com_cx(4), mr1, mr2, mr3, mr4, mres
	REAL    com1( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .		com2( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .		com3( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit,m3loe:m3hie,m3lof:m3hif ),
     .		com4( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit,m4loe:m4hie,m4lof:m4hif ),
     .		 res( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit,m5loe:m5hie,m5lof:m5hif )

* internal variable declarations:
	LOGICAL	TM_FPEQ, first
	INTEGER CX_DIM_LEN, op, i, j, k, l, m, n, icom, idim,
     .		i1, j1, k1, l1, m1, n1, i2, j2, k2, l2, m2, n2, 
     .		i3, j3, k3, l3, m3, n3, i4, j4, k4, l4, m4, n4, 
     .		di1, dj1, dk1, dl1, dm1, dn1, 
     .		di2, dj2, dk2, dl2, dm2, dn2,
     .		di3, dj3, dk3, dl3, dm3, dn3, 
     .		di4, dj4, dk4, dl4, dm4, dn4,
     .		si1, sj1, sk1, sl1, sm1, sn1, 
     .		si2, sj2, sk2, sl2, sm2, sn2,
     .		si3, sj3, sk3, sl3, sm3, sn3, 
     .		si4, sj4, sk4, sl4, sm4, sn4, 
     .		inseed
	REAL	RANDU, RANDN, RANDN2, RHO_UNESCO, THETA_FOFF,
     .		DAYS_FROM_DAY0,
     .          bad1, bad2, bad3, bad4, bad_res, dummy,
     .          op_repl
	REAL*4  R_IN

* equivalence conveniences. Arrays are (idim, icom)
* "inc" is the increments for each axis of each component 
* "lo"  is the starting subscript (minus 1) for each axis of each component
	INTEGER	inc(nferdims,4), lo(nferdims,4)
	EQUIVALENCE
     .	(lo(1,1),si1), (lo(2,1),sj1), (lo(3,1),sk1), (lo(4,1),sl1), (lo(5,1),sm1), (lo(6,1),sn1), 
     .	(lo(1,2),si2), (lo(2,2),sj2), (lo(3,2),sk2), (lo(4,2),sl2), (lo(5,2),sm2), (lo(6,2),sn2), 
     .	(lo(1,3),si3), (lo(2,3),sj3), (lo(3,3),sk3), (lo(4,3),sl3), (lo(5,3),sm3), (lo(6,3),sn3), 
     .	(lo(1,4),si4), (lo(2,4),sj4), (lo(3,4),sk4), (lo(4,4),sl4), (lo(5,4),sm4), (lo(6,4),sn4),
     .	(inc(1,1),di1),(inc(2,1),dj1),(inc(3,1),dk1),(inc(4,1),dl1),(inc(5,1),dm1),(inc(6,1),dn1), 
     .	(inc(1,2),di2),(inc(2,2),dj2),(inc(3,2),dk2),(inc(4,2),dl2),(inc(5,2),dm2),(inc(6,2),dn2), 
     .	(inc(1,3),di3),(inc(2,3),dj3),(inc(3,3),dk3),(inc(4,3),dl3),(inc(5,3),dm3),(inc(6,3),dn3), 
     .	(inc(1,4),di4),(inc(2,4),dj4),(inc(3,4),dk4),(inc(4,4),dl4),(inc(5,4),dm4),(inc(6,4),dn4)

* internal parameter declarations:
	REAL*8          pdays_by_1900
	PARAMETER     ( pdays_by_1900 = 59958230400.0 / (60.*60.*24.) )

* flag(s) for bad or missing values
	bad1 = mr_bad_data ( mr1 )
	bad2 = mr_bad_data ( mr2 )
	bad3 = mr_bad_data ( mr3 )
	bad4 = mr_bad_data ( mr4 )
	bad_res = mr_bad_data( mres )

* For operations, e.g.  IF var GT a THEN b, create a mask of 1's and 0's.
* but for expressions using IFV, the mask needs to be 1's and bad-values.
* If its an IFV expression then use the corresponding THEN and ELSE.

        op = operation
	op_repl = 0.
	IF (ifv) THEN
	   op_repl = bad_res
	   IF (op .EQ. 45) op = 47
	   IF (op .EQ. 46) op = 48
        ENDIF

* compute the delta increment for each axis of each component
	DO icom = 1, num_com
	DO idim = 1, nferdims
	   IF ( CX_DIM_LEN(idim,com_cx(icom)) .EQ. 1 ) THEN
	      inc(idim,icom) = 0
	   ELSE
	      inc(idim,icom) = 1
	   ENDIF
	ENDDO
	ENDDO

* compute the starting subscript for each axis of each component
* (pre-decrement by 1 delta for looping ahead)
	DO icom = 1, num_com
	DO idim = 1, nferdims
	   lo(idim,icom) = cx_lo_ss(com_cx(icom),idim) - inc(idim,icom)
	ENDDO
	ENDDO

* operators, functions, logic structures according to the value of op

* ----------------------------------------------------------------------------
* ------- OPERATORS -------------
*
* "+"
      IF (op .EQ. 1) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	     IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	        res(i,j,k,l,m,n) = bad_res
	     ELSE
	        res(i,j,k,l,m,n) = com1(i1,j1,k1,l1,m1,n1) + com2(i2,j2,k2,l2,m2,n2)
	     ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "-"
      IF (op .EQ. 2) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = com1(i1,j1,k1,l1,m1,n1) - com2(i2,j2,k2,l2,m2,n2)
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF
* "*"

      IF (op .EQ. 3) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = com1(i1,j1,k1,l1,m1,n1) * com2(i2,j2,k2,l2,m2,n2)
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "/"
      IF (op .EQ. 4) THEN
 
	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	     IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2
     .	     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. 0.0   ) THEN
	        res(i,j,k,l,m,n) = bad_res
	     ELSE
	        res(i,j,k,l,m,n) = com1(i1,j1,k1,l1,m1,n1) / com2(i2,j2,k2,l2,m2,n2)
	     ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "^"
      IF (op .EQ. 5) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	     IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2
     .	     .OR.    ( com1(i1,j1,k1,l1,m1,n1) .EQ. 0.0
     .	         .AND. com2(i2,j2,k2,l2,m2,n2) .LT. 0.0 ) ) THEN
	        res(i,j,k,l,m,n) = bad_res
* v*2 compute as v*v, or if integer do as integer version of exponentiation	
	     ELSEIF ( TM_FPEQ(com2(i2,j2,k2,l2,m2,n2), 2.0)  ) THEN
	        res(i,j,k,l,m,n) = com1(i1,j1,k1,l1,m1,n1) * com1(i1,j1,k1,l1,m1,n1)
	     ELSEIF ( com2(i2,j2,k2,l2,m2,n2).EQ.FLOAT(INT(com2(i2,j2,k2,l2,m2,n2))) ) THEN
	        res(i,j,k,l,m,n) = com1(i1,j1,k1,l1,m1,n1) ** INT(com2(i2,j2,k2,l2,m2,n2))
	     ELSEIF ( com1(i1,j1,k1,l1,m1,n1) .LT. 0.0 )	THEN
	        res(i,j,k,l,m,n) = bad_res
	     ELSE
	        res(i,j,k,l,m,n) = com1(i1,j1,k1,l1,m1,n1) ** com2(i2,j2,k2,l2,m2,n2)
	     ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "AND" - logical AND of 2 masks (any non-zero data regarded as .TRUE.)
      IF (op .EQ. 6) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	     IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	        res(i,j,k,l,m,n) = bad_res
	     ELSE
	        IF ( com1(i1,j1,k1,l1,m1,n1).NE.0.0
     .	       .AND. com2(i2,j2,k2,l2,m2,n2).NE.0.0 ) THEN
		   res(i,j,k,l,m,n) = 1.0
	        ELSE
	  	 res(i,j,k,l,m,n) = op_repl
	        ENDIF
	     ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "OR" - logical OR of 2 masks (any non-zero data regarded as .TRUE.)

      IF (op .EQ. 7) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	     IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	        res(i,j,k,l,m,n) = bad_res
	     ELSE
	        IF ( com1(i1,j1,k1,l1,m1,n1).NE.0.0
     .	        .OR. com2(i2,j2,k2,l2,m2,n2).NE.0.0 ) THEN
	  	   res(i,j,k,l,m,n) = 1.0
	        ELSE
		   res(i,j,k,l,m,n) = op_repl
	        ENDIF
	     ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "GT" - mask of 1's where grid 1 exceeds grid 2, 0's elsewhere
* "GT" - mask of 1's where grid 1 exceeds grid 2, bad-val elsewhere
      IF (op .EQ. 8) THEN

 	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	     IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	        res(i,j,k,l,m,n) = bad_res
	     ELSE
	      IF ( com1(i1,j1,k1,l1,m1,n1) .GT. com2(i2,j2,k2,l2,m2,n2) ) THEN
	  	   res(i,j,k,l,m,n) = 1.0
	        ELSE
		   res(i,j,k,l,m,n) = op_repl
	        ENDIF
	     ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "GE" - mask of 1's where grid 1 exceeds or equals grid 2, 0's elsewhere
* "GE" - mask of 1's where grid 1 exceeds or equals grid 2, bad elsewhere

      IF (op .EQ. 9) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1,m1,n1) .GE. com2(i2,j2,k2,l2,m2,n2) ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = op_repl
	      ENDIF
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "LT" - mask of 1's where grid 1 is less than grid 2, 0's elsewhere
* "LT" - mask of 1's where grid 1 is less than grid 2, bad elsewhere
      IF (op .EQ. 10) THEN

 	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1,m1,n1) .LT. com2(i2,j2,k2,l2,m2,n2) ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = op_repl
	      ENDIF
	   ENDIF
 
 	     ENDDO
 	    ENDDO
 	   ENDDO
 	   ENDDO
	  ENDDO
	ENDDO

	RETURN

      ENDIF

* "LE" - mask of 1's where grid 1 is less than or equal to grid 2, 0's elsewhere
* "LE" - mask of 1's where grid 1 is less than or equal to grid 2, bad elsewhere
      IF (op .EQ. 11) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1,m1,n1) .LE. com2(i2,j2,k2,l2,m2,n2) ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = op_repl
	      ENDIF
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "EQ" - mask of 1's where grids are equal, 0's elsewhere
* "EQ" - mask of 1's where grids are equal, bad elsewhere

      IF (op .EQ. 12) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2
	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. com2(i2,j2,k2,l2,m2,n2) ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = op_repl
	      ENDIF
	   ENDIF



	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO


	RETURN

      ENDIF

* "NE" - mask of 1's where grids are not equal, 0's elsewhere
      IF (op .EQ. 13) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      IF ( com1(i1,j1,k1,l1,m1,n1) .NE. com2(i2,j2,k2,l2,m2,n2) ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = op_repl
	      ENDIF
	   ENDIF



	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO


	RETURN

      ENDIF

      IF (op.EQ.14 .OR.op.EQ.15 .OR.op.EQ.16 .OR.op.EQ.17 .OR.op.EQ.18) THEN
	STOP 'OPERATOR NOT IMPLEMENTED'
      ENDIF

* -----------------------------------------------------------------------------
* -------- FUNCTIONS ------------

* "EXP" - exponential function 
      IF (op .EQ. 19) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1
     .     .OR. com1(i,j,k,l,m,n) .GT. 77.0 ) THEN    ! 8/92
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = EXP(com1(i,j,k,l,m,n))
	   ENDIF



	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO


	RETURN

      ENDIF

* "LOG"
      IF (op .EQ. 20) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1
     .    .OR.  com1(i,j,k,l,m,n) .LE. 0.0 ) THEN    ! 8/92
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = LOG10( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "MAX"
      IF (op .EQ. 21) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = MAX( com1(i1,j1,k1,l1,m1,n1), com2(i2,j2,k2,l2,m2,n2) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "MIN"

      IF (op .EQ. 22) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = MIN( com1(i1,j1,k1,l1,m1,n1), com2(i2,j2,k2,l2,m2,n2) )
	   ENDIF


	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "INT"
      IF (op .EQ. 23) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = INT( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "ABS"
      IF (op .EQ. 24) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = ABS( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "SIN"
      IF (op .EQ. 25) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = SIN( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "COS"
      IF (op .EQ. 26) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = COS( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "TAN"
      IF (op .EQ. 27) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = TAN( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "LN"
      IF (op .EQ. 28) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1
     .    .OR.  com1(i,j,k,l,m,n) .LE. 0.0 ) THEN    ! 8/92
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = LOG( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "MOD"
      IF (op .EQ. 29) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = MOD( com1(i1,j1,k1,l1,m1,n1), com2(i2,j2,k2,l2,m2,n2) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "MISSING" - replace missing value flag with values in grid 2
      IF (op .EQ. 30) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .NE. bad1 ) THEN
              res(i,j,k,l,m,n) = com1(i1,j1,k1,l1,m1,n1)
           ELSEIF ( com2(i2,j2,k2,l2,m2,n2) .NE. bad2 ) THEN
              res(i,j,k,l,m,n) = com2(i2,j2,k2,l2,m2,n2)
           ELSE
              res(i,j,k,l,m,n) = bad_res
           ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "IGNORE0" - replace zeros with bad flags
      IF (op .EQ. 31) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSEIF ( com1(i,j,k,l,m,n) .EQ. 0 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = com1(i,j,k,l,m,n)
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "ATAN"
      IF (op .EQ. 32) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = ATAN( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "ATAN2" - 2 argument arc tangent
      IF (op .EQ. 33) THEN

	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .	   .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSEIF ( com1(i1,j1,k1,l1,m1,n1) .EQ. 0.0
     .	   .AND.    com2(i2,j2,k2,l2,m2,n2) .EQ. 0.0 ) THEN
	      res(i,j,k,l,m,n) = bad_res	! indeterminant
	   ELSE
	      res(i,j,k,l,m,n) = ATAN2( com1(i1,j1,k1,l1,m1,n1),com2(i2,j2,k2,l2,m2,n2) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "ASIN"
      IF (op .EQ. 34) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1
     .     .OR. com1(i,j,k,l,m,n) .LT. -1.
     .     .OR. com1(i,j,k,l,m,n) .GT. +1. ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = ASIN( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "ACOS"
      IF (op .EQ. 35) THEN

	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1
     .     .OR. com1(i,j,k,l,m,n) .LT. -1.
     .     .OR. com1(i,j,k,l,m,n) .GT. +1. ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      res(i,j,k,l,m,n) = ACOS( com1(i,j,k,l,m,n) )
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "RANDU" - random uniform
* use the first value as the seed
      IF (op .EQ. 36) THEN

	first = .TRUE.
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSEIF ( first ) THEN
* ... note: floating point representation is passed as seed integer
              R_IN = REAL(com1(i,j,k,l,m,n))  ! for possibly double-prec. arg
	      res(i,j,k,l,m,n) = RANDU( R_IN )
	      first = .FALSE.
	   ELSE
              R_IN = 0.
	      res(i,j,k,l,m,n) = RANDU(R_IN)			!kob 12/96
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "RANDN" - random normal
* use the first value as the seed
      IF (op .EQ. 37) THEN

	first = .TRUE.
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSEIF ( first ) THEN
* ... note: floating point representation is passed as seed integer
              R_IN = REAL(com1(i,j,k,l,m,n))  ! for possibly double-prec. arg
	      res(i,j,k,l,m,n) = RANDN( R_IN )
	      first = .FALSE.
	   ELSE
              R_IN = 0.
	      res(i,j,k,l,m,n) = RANDN(R_IN)			!kob 12/96
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "RHO" - UNESCO equation of state:
* rho = rho(salinity, temperature, pressure)
* three components - each may be a different XYZT shape
      IF (op .EQ. 38) THEN

	n1 = sn1
	n2 = sn2
	n3 = sn3
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	n1 = n1 + dn1
	n2 = n2 + dn2
	n3 = n3 + dn3
	m1 = sm1
	m2 = sm2
	m3 = sm3
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	 m1 = m1 + dm1
	 m2 = m2 + dm2
	 m3 = m3 + dm3
	  l1 = sl1
	  l2 = sl2
	  l3 = sl3
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  l3 = l3 + dl3
	  k1 = sk1
	  k2 = sk2
	  k3 = sk3
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   k3 = k3 + dk3
	   j1 = sj1
	   j2 = sj2
	   j3 = sj3
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    j3 = j3 + dj3
	    i1 = si1
	    i2 = si2
	    i3 = si3
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2
	     i3 = i3 + di3

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2
     .     .OR. com3(i3,j3,k3,l3,m3,n3) .EQ. bad3
     .     .OR. com1(i1,j1,k1,l1,m1,n1) .LT. 0.0   ) THEN  ! salt>0 8/92
              res(i,j,k,l,m,n) = bad_res
           ELSE
              res(i,j,k,l,m,n) = RHO_UNESCO( com1(i1,j1,k1,l1,m1,n1),
     .                                   com2(i2,j2,k2,l2,m2,n2),
     .                                   com3(i3,j3,k3,l3,m3,n3) )
           ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "THETA" - potential temperature from BRYDEN,H.,1973,DEEP-SEA RES
*           and FOFONOFF,N,M,1977,DEEP-SEA RES
* theta = theta(salinity, temperature, pressure,reference_pressure)
* three components - each may be a different XYZT shape
      IF (op .EQ. 39) THEN

	n1 = sn1
	n2 = sn2
	n3 = sn3
	n4 = sn4
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	n1 = n1 + dn1
	n2 = n2 + dn2
	n3 = n3 + dn3
	n4 = n4 + dn4
	m1 = sm1
	m2 = sm2
	m3 = sm3
	m4 = sm4
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	 m1 = m1 + dm1
	 m2 = m2 + dm2
	 m3 = m3 + dm3
	 m4 = m4 + dm4
	  l1 = sl1
	  l2 = sl2
	  l3 = sl3
	  l4 = sl4
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  l3 = l3 + dl3
	  l4 = l4 + dl4
	  k1 = sk1
	  k2 = sk2
	  k3 = sk3
	  k4 = sk4
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   k3 = k3 + dk3
	   k4 = k4 + dk4
	   j1 = sj1
	   j2 = sj2
	   j3 = sj3
	   j4 = sj4
	  DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  j3 = j3 + dj3
	  j4 = j4 + dj4
	  i1 = si1
	  i2 = si2
	  i3 = si3
	  i4 = si4
	   DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2
	   i3 = i3 + di3
	   i4 = i4 + di4

	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2
     .     .OR. com3(i3,j3,k3,l3,m3,n3) .EQ. bad3
     .     .OR. com4(i4,j4,k4,l4,m4,n4) .EQ. bad4 ) THEN
              res(i,j,k,l,m,n) = bad_res
           ELSE
              res(i,j,k,l,m,n) = THETA_FOFF( com1(i1,j1,k1,l1,m1,n1),
     .                                   com2(i2,j2,k2,l2,m2,n2),
     .                                   com3(i3,j3,k3,l3,m3,n3),
     .                                   com4(i4,j4,k4,l4,m4,n4) )
           ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "DAYS1900(year, month, day)" - number of days since 1900
* three components - each may be a different XYZT shape
* *** note: returns days since 1-jan-1900, on the STANDARD CALENDAR

      IF (op .EQ. 40) THEN

	n1 = sn1
	n2 = sn2
	n3 = sn3
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	n1 = n1 + dn1
	n2 = n2 + dn2
	n3 = n3 + dn3
	m1 = sm1
	m2 = sm2
	m3 = sm3
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	 m1 = m1 + dm1
	 m2 = m2 + dm2
	 m3 = m3 + dm3
	  l1 = sl1
	  l2 = sl2
	  l3 = sl3
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  l3 = l3 + dl3
	  k1 = sk1
	  k2 = sk2
	  k3 = sk3
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   k3 = k3 + dk3
	   j1 = sj1
	   j2 = sj2
	   j3 = sj3
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    j3 = j3 + dj3
	    i1 = si1
	    i2 = si2
	    i3 = si3
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2
	     i3 = i3 + di3

* 4/98, 2nd references to com2, com3 incorrectly used i1,etc for array references
*       fixed that up - kob
	   IF ( com1(i1,j1,k1,l1,m1,n1) .EQ. bad1
     .     .OR. com2(i2,j2,k2,l2,m2,n2) .EQ. bad2
     .     .OR. com3(i3,j3,k3,l3,m3,n3) .EQ. bad3
     .     .OR. com2(i2,j2,k2,l2,m2,n2) .GT. 12.0
     .     .OR. com2(i2,j2,k2,l2,m2,n2) .LT. 1.0
     .     .OR. com3(i3,j3,k3,l3,m3,n3) .LT. 0.0    ) THEN
              res(i,j,k,l,m,n) = bad_res
           ELSE
              res(i,j,k,l,m,n) = DAYS_FROM_DAY0( pdays_by_1900,
     .					     INT(com1(i1,j1,k1,l1,m1,n1)),
     .                                       INT(com2(i2,j2,k2,l2,m2,n2)),
     .                                       INT(com3(i3,j3,k3,l3,m3,n3)),
     .                                       dummy)
           ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF


* "RANDU2" - random uniform 
* Call the gfortran RANDOM_SEED and RANDOM_SEED functions.
* See ticket 1886 for discussions of RANDU
*
* iseed = 0   use previous seed - don't reinitialize
* iseed = -1 initialize using system clock
* iseed = integer > 0 initialize using that integer: results will be repeatable

      IF (op .EQ. 41) THEN

	CONTINUE
	n2 = sn2
	m2 = sm2
	l2 = sl2
	k2 = sk2
	j2 = sj2
	i2 = si2
        inseed = INT(com2(i2,j2,k2,l2,m2,n2))
        CALL INIT_RANDOM_SEED(inseed)
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSE
	      CALL RANDOM_NUMBER(res(i,j,k,l,m,n))
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "RANDN2" - random normal using gfortran RANDOM_SEED and RANDOM_SEED functions.
* seed set as for RANDU2
      IF (op .EQ. 42) THEN

	n2 = sn2
	m2 = sm2
	l2 = sl2
	k2 = sk2
	j2 = sj2
	i2 = si2
	first = .TRUE.
        inseed = INT(com2(i2,j2,k2,l2,m2,n2))
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	   IF ( com1(i,j,k,l,m,n) .EQ. bad1 ) THEN
	      res(i,j,k,l,m,n) = bad_res
	   ELSEIF ( first ) THEN
	      res(i,j,k,l,m,n) = RANDN2( inseed )
	      first = .FALSE.
	   ELSE
	      res(i,j,k,l,m,n) = RANDN2(0)
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

      IF (op .EQ. 42) THEN

	STOP 'FUNCTION NOT IMPLEMENTED'

      ENDIF
* ----------------------------------------------------------------------------
* ------- LOGIC STRUCTURES -------------
*
      IF (op .EQ. 44 .OR. op.EQ.45) THEN

* "THEN" - comes from IF (condition) THEN (choice)   ... no "ELSE" given
* component 1 is the condition.  component 2 is the choice
	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .NE. bad1
     .	  .AND. com1(i1,j1,k1,l1,m1,n1) .NE. 0.0
     .	  .AND. com2(i2,j2,k2,l2,m2,n2) .NE. bad2	 ) THEN
	      res(i,j,k,l,m,n) = com2(i2,j2,k2,l2,m2,n2)
	   ELSE
	      res(i,j,k,l,m,n) = bad_res
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF
* "ELSE" - comes from IF (condition) THEN (choice1) ELSE (choice2)
* component 1 is the condition.  components 2 and 3 are the choices

      IF (op .EQ. 46) THEN
	n1 = sn1
	n2 = sn2
	n3 = sn3
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	n1 = n1 + dn1
	n2 = n2 + dn2
	n3 = n3 + dn3
	m1 = sm1
	m2 = sm2
	m3 = sm3
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	 m1 = m1 + dm1
	 m2 = m2 + dm2
	 m3 = m3 + dm3
	  l1 = sl1
	  l2 = sl2
	  l3 = sl3
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  l3 = l3 + dl3
	  k1 = sk1
	  k2 = sk2
	  k3 = sk3
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   k3 = k3 + dk3
	   j1 = sj1
	   j2 = sj2
	   j3 = sj3
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    j3 = j3 + dj3
	    i1 = si1
	    i2 = si2
	    i3 = si3
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2
	     i3 = i3 + di3

	   IF ( com1(i1,j1,k1,l1,m1,n1) .NE. bad1
     .	  .AND. com1(i1,j1,k1,l1,m1,n1) .NE. 0.0 ) THEN
	      IF ( com2(i2,j2,k2,l2,m2,n2) .NE. bad2 ) THEN
	         res(i,j,k,l,m,n) = com2(i2,j2,k2,l2,m2,n2)
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
	      ENDIF
	   ELSE
	      IF ( com3(i3,j3,k3,l3,m3,n3) .NE. bad3 ) THEN
	         res(i,j,k,l,m,n) = com3(i3,j3,k3,l3,m3,n3)
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
	      ENDIF
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

* "THEN" - comes from IFV (condition) THEN (choice)   ... no "ELSE" given
* component 1 is the condition.  component 2 is the choice

      IF (op .EQ. 47) THEN
	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  k1 = sk1
	  k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   j1 = sj1
	   j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

	   IF ( com1(i1,j1,k1,l1,m1,n1) .NE. bad1
     .	  .AND. com2(i2,j2,k2,l2,m2,n2) .NE. bad2	 ) THEN
	      res(i,j,k,l,m,n) = com2(i2,j2,k2,l2,m2,n2)
	   ELSE
	      res(i,j,k,l,m,n) = bad_res
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF
* "ELSE" - comes from IFV (condition) THEN (choice1) ELSE (choice2)
* component 1 is the condition.  components 2 and 3 are the choices

      IF (op .EQ. 48) THEN
	n1 = sn1
	n2 = sn2
	n3 = sn3
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	n1 = n1 + dn1
	n2 = n2 + dn2
	n3 = n3 + dn3
	m1 = sm1
	m2 = sm2
	m3 = sm3
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	 m1 = m1 + dm1
	 m2 = m2 + dm2
	 m3 = m3 + dm3
	  l1 = sl1
	  l2 = sl2
	  l3 = sl3
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  l1 = l1 + dl1
	  l2 = l2 + dl2
	  l3 = l3 + dl3
	  k1 = sk1
	  k2 = sk2
	  k3 = sk3
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   k1 = k1 + dk1
	   k2 = k2 + dk2
	   k3 = k3 + dk3
	   j1 = sj1
	   j2 = sj2
	   j3 = sj3
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    j3 = j3 + dj3
	    i1 = si1
	    i2 = si2
	    i3 = si3
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2
	     i3 = i3 + di3

	   IF ( com1(i1,j1,k1,l1,m1,n1) .NE. bad1 ) THEN
	      IF ( com2(i2,j2,k2,l2,m2,n2) .NE. bad2 ) THEN
	         res(i,j,k,l,m,n) = com2(i2,j2,k2,l2,m2,n2)
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
	      ENDIF
	   ELSE
	      IF ( com3(i3,j3,k3,l3,m3,n3) .NE. bad3 ) THEN
	         res(i,j,k,l,m,n) = com3(i3,j3,k3,l3,m3,n3)
	      ELSE
	         res(i,j,k,l,m,n) = bad_res
	      ENDIF
	   ENDIF

	    ENDDO
	   ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN

      ENDIF

	END
