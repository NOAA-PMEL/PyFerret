      SUBROUTINE STRING_OPERATE (operation, com_cx,
     .				 com1, mr1,
     .				 com2, mr2,
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
* perform a logical string operation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V533 *sh* 7/01 - based upon OPERATE.F
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER	operation, com_cx(4), mr1, mr2, mres, m, n
	REAL*8  com1( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .		com2( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )
        REAL    res( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit,m5loe:m5hie,m5lof:m5hif )

* internal variable declarations:
	INTEGER CX_DIM_LEN, COMPARE_C_STRINGS,
     .          i, j, k, l, icom, idim, comp, op,
     .		i1, j1, k1, l1, m1, n1, i2, j2, k2, l2, m2, n2,
     .		di1, dj1, dk1, dl1, dm1, dn1, 
     .		di2, dj2, dk2, dl2, dm2, dn2,
     .		si1, sj1, sk1, sl1, sm1, sn1, 
     .		si2, sj2, sk2, sl2, sm2, sn2

* internal parameter declaration
        INTEGER num_com
        PARAMETER (num_com = 2)

* equivalence conveniences. Arrays are (idim, icom)
* "inc" is the increments for each axis of each component
* "lo"  is the starting subscript (minus 1) for each axis of each component
	INTEGER	inc(nferdims,4), lo(nferdims,4)
	EQUIVALENCE
     .	(lo(1,1),si1), (lo(2,1),sj1), (lo(3,1),sk1), (lo(4,1),sl1), (lo(5,1),sm1), (lo(6,1),sn1), 
     .	(lo(1,2),si2), (lo(2,2),sj2), (lo(3,2),sk2), (lo(4,2),sl2), (lo(5,2),sm2), (lo(6,2),sn2), 
     .	(inc(1,1),di1),(inc(2,1),dj1),(inc(3,1),dk1),(inc(4,1),dl1),(inc(5,1),dm1),(inc(6,1),dn1), 
     .  (inc(1,2),di2),(inc(2,2),dj2),(inc(3,2),dk2),(inc(4,2),dl2),(inc(5,2),dm2),(inc(6,2),dn2)

* compute the delta increment for each axis of) each component
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

* operator number (8 thru 13 OK)
        op = operation - 7
        IF (op.LT.1 .OR. op.GT.6) STOP 'STR OP?'

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

             comp = COMPARE_C_STRINGS(com1(i1,j1,k1,l1,m1,n1),
     .                                com2(i2,j2,k2,l2,m2,n2) )

             GOTO (10,20,30,40,50,60) op

* op = 1 "GT"
 10          IF ( comp .GT. 0 ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = 0.0
	      ENDIF
              CYCLE

* op = 2 "GE"
 20          IF ( comp .GE. 0 ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = 0.0
	      ENDIF
              CYCLE

* op = 3 "LT"
 30          IF ( comp .LT. 0 ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = 0.0
	      ENDIF
              CYCLE

* op = 4 "LE"
 40          IF ( comp .LE. 0 ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = 0.0
	      ENDIF
              CYCLE

* op = 5 "EQ"
 50          IF ( comp .EQ. 0 ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = 0.0
	      ENDIF
              CYCLE

* op = 6 "NE"
 60          IF ( comp .NE. 0 ) THEN
		 res(i,j,k,l,m,n) = 1.0
	      ELSE
		 res(i,j,k,l,m,n) = 0.0
	      ENDIF

	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

	RETURN
        END
