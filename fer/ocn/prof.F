      SUBROUTINE PROF( KMIX,	UK,  muk,
     .				VK,  mvk,
     .				ROK, mrok,
     .				RNUE,mrnue, HNUE,mhnue  )	! 1522 mod

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/7/86	extracted from SUBROUTINE HDIFFUS
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* revision 0.1 - 5/7/87 - changes in IF tests to avoid under/over flow
* V200:  7/25/89 - 4D symmetrical
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl.parm'		! parameter definitions
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xonedim.cmn'	! geometric constants
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'		! parameter definitions
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'	! geometric constants
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	KMIX, muk, mvk, mrok, mrnue, mhnue
* subscript ranges from memory variable table ...
	REAL      uk( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		  vk( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		 rok( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .		rnue( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit ),
     .		hnue( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit )

* internal variable declarations:
	INTEGER	i, j, k, l
	REAL	bad_uk, bad_vk, bad_ro, bad_rnue, bad_hnue,
     .		fricmax, gr, di, windmix

      DATA FRICMAX / 50./,GR/980./					! 1541

* flag(s) for bad or missing values
	bad_uk = mr_bad_data ( muk  )
	bad_vk = mr_bad_data ( mvk  )
	bad_ro = mr_bad_data ( mrok )
	bad_rnue = mr_bad_data ( mrnue )   ! added 2/93
	bad_hnue = mr_bad_data ( mhnue )   ! added 2/93

	DO 1000 l = mr_lo_s4(mrnue), mr_hi_s4(mrnue)
	DO 1000 k = mr_lo_s3(mrnue), mr_hi_s3(mrnue)

	IF ( k .LT. kmix ) THEN 	! kmix may be generalized to im x jm

	DO 50 j = mr_lo_s2(mrnue), mr_hi_s2(mrnue)
	DO 50 i = mr_lo_s1(mrnue), mr_hi_s1(mrnue)

	IF ( uk  (i,j,k,l) .EQ. bad_uk
     .	.OR. vk  (i,j,k,l) .EQ. bad_vk
     .	.OR. rok (i,j,k,l) .EQ. bad_ro ) THEN
	   rnue(i,j,k,l) = bad_rnue
	   hnue(i,j,k,l) = bad_hnue
	   GOTO 50
	ENDIF

      HNUE(i,j,k,l)=((Uk(I,j,K+1,l)-Uk(i,j,k,l))**2
     .		   + (Vk(I,j,k+1,l)-Vk(i,j,k,l))**2)	! 1545 mod
C									! 1547
C     COMPUTE RICHARDSON NUMBER						! 1548
C									! 1549
	di = dzz(k+1)		! ref.   1088   DI(I,K)=ZDZZ(K+1)-ZDZZ(K)
	HNUE(i,j,k,l)= GR*DI*(ROk(I,j,k+1,l)-ROk(i,j,k,l))
     .		     /(HNUE(i,j,k,l)+1.E-20)! 1552 mod

C									! 1553
C     NEG RICH NUMBERS WILL BE HANDLED BY CONVECTIVE ADJUSTMENT!	! 1554
C									! 1555
	hnue(i,j,k,l) = ABS( hnue(i,j,k,l) )			     ! 1556 mod

* under/over flow fix
	IF ( hnue(i,j,k,l) .GT. 1.E11 ) THEN
	 rnue(i,j,k,l) = bam
	 hnue(i,j,k,l) = bah
	 GOTO 50
	ENDIF

      HNUE(i,j,k,l)=1.+5.*HNUE(i,j,k,l)					! 1559
C									! 1561
C     COMPUTE VERTICAL VISCOSITY					! 1562
C									! 1563
      RNUE(i,j,k,l) = FRICMAX / HNUE(i,j,k,l)**2		! 1566 mod
C									! 1567
C     COMPUTE VERTICAL DIFFUSIVITY					! 1568
C									! 1569
      HNUE(i,j,k,l)=RNUE(i,j,k,l)/HNUE(i,j,k,l)+BAH		! 1572 mod
C									! 1573
C     ADD IN BACKGROUND							! 1574
C									! 1575
      RNUE(i,j,k,l)=RNUE(i,j,k,l)+BAM				! 1578 mod
 50	CONTINUE

	ELSE

C									! 1579
C     SET BACKGROUND VALUES BELOW LEVEL KMIX				! 1580
C									! 1581
	DO 60 j = mr_lo_s2(mrnue), mr_hi_s2(mrnue)
	DO 60 i = mr_lo_s1(mrnue), mr_hi_s1(mrnue)		! 1583 mod 
      RNUE(i,j,k,l)=BAM							! 1584
  60  HNUE(i,j,k,l)=BAH							! 1585

	ENDIF
C									! 1586
C     APPROX HIGH FREQ WIND MIXING NEAR SURFACE				! 1587
C									! 1588
	IF ( k .EQ. 1 ) THEN
      WINDMIX=10.							! 1589
	DO 70 j = mr_lo_s2(mrnue), mr_hi_s2(mrnue)
	DO 70 i = mr_lo_s1(mrnue), mr_hi_s1(mrnue)		! 1590 mod 
      IF (   rnue(i,j,k,l) .NE. bad_rnue
     . .AND. RNUE(i,j,k,l) .LT. WINDMIX ) RNUE(i,j,k,l)=WINDMIX	! 1591 mod
      IF (   hnue(i,j,k,l) .NE. bad_hnue
     . .AND. HNUE(i,j,k,l) .LT. WINDMIX ) HNUE(i,j,k,l)=WINDMIX	! 1592 mod
70    CONTINUE								! 1593
	ENDIF

 1000	CONTINUE

      RETURN								! 1594
      END								! 1595

