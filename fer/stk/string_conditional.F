      SUBROUTINE STRING_CONDITIONAL (operation, num_com, com_cx,
     .				 com1, mr1,
     .				 com2, mr2,
     .				 com3, mr3,
     .				 res, mres )

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
* perform a conditional choice on string variables
!   IF cond THEN string1_array ELSE string2_array

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V533 *sh* 7/01 - based upon OPERATE.F
* V67  *acm 3/11 - Fix bugs 1764,1795 (copy/paste errors in setting up 3rd component)
* V671 7/11  *acm* Ticket 1868 IFV - masking expressions with ifValid, treating 
*		   zero as valid. 
*		   (String expressions with IFV are identical to behavior with IF.)
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V751 11/19 *acm* fix copy-paste bug in ELSE
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER	operation, num_com, com_cx(4), mr1, mr2, mr3, mres, i3, n2
	REAL    com1( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )
        REAL*8	com2( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .  	com3( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit,m3loe:m3hie,m3lof:m3hif ),
     .  	res( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit,m5loe:m5hie,m5lof:m5hif )

* internal variable declarations:
	INTEGER CX_DIM_LEN,
     .          i, j, k, l, m, n, icom, idim,
     .		i1, j1, k1, l1, m1, n1, i2, j2, k2, l2, m2, n2
     .		i3, j3, k3, l3, m3, n3,
     .		di1, dj1, dk1, dl1, dm1, dn1, 
     .		di2, dj2, dk2, dl2, dm2, dn2,
     .		di3, dj3, dk3, dl3, dm3, dn3,
     .		si1, sj1, sk1, sl1, sm1, sn1, 
     .		si2, sj2, sk2, sl2, sm2, sn2,
     .		si3, sj3, sk3, sl3, sm3, sn3, 
     .          op
        REAL    bad1

* equivalence conveniences. Arrays are (idim, icom)
* "inc" is the increments for each axis of each component
* "lo"  is the starting subscript (minus 1) for each axis of each component
	INTEGER	inc(nferdims,4), lo(nferdims,4)
	EQUIVALENCE
     .	(lo(1,1),si1), (lo(2,1),sj1), (lo(3,1),sk1), (lo(4,1),sl1), (lo(5,1),sm1), (lo(6,1),sn1), 
     .	(lo(1,2),si2), (lo(2,2),sj2), (lo(3,2),sk2), (lo(4,2),sl2), (lo(5,2),sm2), (lo(6,2),sn2), 
     .	(lo(1,3),si3), (lo(2,3),sj3), (lo(3,3),sk3), (lo(4,3),sl3), (lo(5,3),sm3), (lo(6,3),sn3),
     .	(inc(1,1),di1),(inc(2,1),dj1),(inc(3,1),dk1),(inc(4,1),dl1),(inc(5,1),dm1),(inc(6,1),dn1), 
     .	(inc(1,2),di2),(inc(2,2),dj2),(inc(3,2),dk2),(inc(4,2),dl2),(inc(5,2),dm2),(inc(6,2),dn2), 
     .	(inc(1,3),di3),(inc(2,3),dj3),(inc(3,3),dk3),(inc(4,3),dl3),(inc(5,3),dm3),(inc(6,3),dn3)

* flag for bad or missing values
        bad1 = mr_bad_data ( mr1 )

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

* operator number (1 thru 3 OK)
        op = operation
        IF (op .EQ. 5) op = 3  ! IFV works as for IF
        IF (op.LT.1 .OR. op.GT.3) STOP 'STR OP?'

        IF (op .LT. 3) THEN


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
     .	  .AND. com1(i1,j1,k1,l1,m1,n1) .NE. 0.0       ) THEN
              CALL COPY_C_STRING(com2(i2,j2,k2,l2,m2,n2), res(i,j,k,l,m,n))
           ELSE
              CALL SET_NULL_C_STRING(res(i,j,k,l,m,n))
           ENDIF

	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

        ELSE !  op is 3 for ELSE 

* "ELSE" - comes from IF (condition) THEN (choice1) ELSE (choice2)
* component 1 is the condition.  components 2 and 3 are the choices
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
     .	  .AND. com1(i1,j1,k1,l1,m1,n1) .NE. 0.0       ) THEN
              CALL COPY_C_STRING(com2(i2,j2,k2,l2,m2,n2), res(i,j,k,l,m,n))
           ELSE
              CALL COPY_C_STRING(com3(i3,j3,k3,l3,m3,n3), res(i,j,k,l,m,n))
           ENDIF

	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	ENDIF

	RETURN

        END
