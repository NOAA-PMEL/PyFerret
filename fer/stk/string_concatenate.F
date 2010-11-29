      SUBROUTINE STRING_CONCATENATE (com_cx,
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
* perform a string concatenation operation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V533 *sh* 7/01 - based upon OPERATE.F

	include	'ferret.parm'
	include	'errmsg.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER	com_cx(4), mr1, mr2, mres
	REAL*8  com1( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		com2( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )
        REAL*8   res( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit )

* internal variable declarations:
	INTEGER CX_DIM_LEN, COMPARE_C_STRINGS,
     .          i, j, k, l, icom, idim,
     .		i1, j1, k1, l1, i2, j2, k2, l2,
     .		di1, dj1, dk1, dl1, di2, dj2, dk2, dl2,
     .		si1, sj1, sk1, sl1, si2, sj2, sk2, sl2

* internal parameter declaration
        INTEGER num_com
        PARAMETER (num_com = 2)

* equivalence conveniences
* "inc" is the increments for each axis of each component
* "lo"  is the starting subscript (minus 1) for each axis of each component
	INTEGER	inc(4,4), lo(4,4)
	EQUIVALENCE
     .	(lo(1,1),si1), (lo(2,1),sj1), (lo(3,1),sk1), (lo(4,1),sl1),
     .	(lo(1,2),si2), (lo(2,2),sj2), (lo(3,2),sk2), (lo(4,2),sl2),
     .	(inc(1,1),di1),(inc(2,1),dj1),(inc(3,1),dk1),(inc(4,1),dl1),
     .	(inc(1,2),di2),(inc(2,2),dj2),(inc(3,2),dk2),(inc(4,2),dl2)

* compute the delta increment for each axis of each component
	DO 2 icom = 1, num_com
	DO 2 idim = 1, 4
	   IF ( CX_DIM_LEN(idim,com_cx(icom)) .EQ. 1 ) THEN
	      inc(idim,icom) = 0
	   ELSE
	      inc(idim,icom) = 1
	   ENDIF
 2	CONTINUE

* compute the starting subscript for each axis of each component
* (pre-decrement by 1 delta for looping ahead)
	DO 4 icom = 1, num_com
	DO 4 idim = 1, 4
 4	lo(idim,icom) = cx_lo_ss(com_cx(icom),idim) - inc(idim,icom)


	l1 = sl1
	l2 = sl2
	DO 200 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	k1 = sk1
	k2 = sk2
	 DO 200 k = mr_lo_s3(mres), mr_hi_s3(mres)
	 k1 = k1 + dk1
	 k2 = k2 + dk2
	 j1 = sj1
	 j2 = sj2
	  DO 200 j = mr_lo_s2(mres), mr_hi_s2(mres)
	  j1 = j1 + dj1
	  j2 = j2 + dj2
	  i1 = si1
	  i2 = si2
	   DO 200 i = mr_lo_s1(mres), mr_hi_s1(mres)
	   i1 = i1 + di1
	   i2 = i2 + di2

             CALL CONCATENATE_C_STRINGS(com1(i1,j1,k1,l1),
     .                                  com2(i2,j2,k2,l2),
     .                                  res(i,j,k,l)      )

 200	CONTINUE

	RETURN
        END
