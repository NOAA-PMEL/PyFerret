	SUBROUTINE KNUDSEN( temp, mtemp, salt, msalt, dens, mdens )

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
* lifted from COX routine KNUDSEN to compute the density of sea water via
* the Knudsen formula 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 12/15/87
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include 'ferret.parm'
	include 'gfdl.parm'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xonedim.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'
#endif

* calling argument declarations:
	INTEGER	   mtemp, msalt, mdens
	REAL    temp( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		salt( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		dens( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	REAL T3,S2,T2,S3,F1,F2,F3,FS,SIGMA,A,B1,B2,B,CO,ALPHA,ALPSTD
	REAL S,T,DN,D
	REAL POTEM
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAL	bad_temp, bad_salt, bad_dens

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(mdens)
	i_hi = mr_hi_s1(mdens)
	j_lo = mr_lo_s2(mdens)
	j_hi = mr_hi_s2(mdens)

* flag for bad/missing data
	bad_temp = mr_bad_data( mtemp )
	bad_salt = mr_bad_data( msalt )
	bad_dens = mr_bad_data( mdens )

	DO 2000 l = mr_lo_s4(mdens), mr_hi_s4(mdens)
	DO 2000 k = mr_lo_s3(mdens), mr_hi_s3(mdens)

* depth in meters
	D = zdzz(k) / 100.

* main loop - compute one horizontal grid
	DO 100 j = j_lo, j_hi
	DO 100 i = i_lo, i_hi

	   IF ( temp(i,j,k,l).EQ.bad_temp .OR. salt(i,j,k,l).EQ.bad_salt ) THEN
	      dens(i,j,k,l) = bad_dens
	      GOTO 100
	   ENDIF

* true salinity(ppt) and potential temperature
	   s = (salt(i,j,k,l)*1000.D0) + 35.
	   t = POTEM( temp(i,j,k,l), s, d )

* stolen KNUDSEN code
      T2 = T*T
      T3= T2* T
      S2 = S*S
      S3 = S2 * S
      F1 = -(T-3.98)**2 * (T+283.)/(503.57*(T+67.26))
      F2 = T3*1.0843E-6 - T2*9.8185E-5 + T*4.786E-3
      F3 = T3*1.667E-8 - T2*8.164E-7 + T*1.803E-5
      FS= S3*6.76786136D-6 - S2*4.8249614D-4 + S*8.14876577D-1
      SIGMA= F1 + (FS+3.895414D-2)*(1.-F2+F3*(FS-.22584586D0))
      A= D*1.0E-4*(105.5+ T*9.50 - T2*0.158 - D*T*1.5E-4)  -
     1(227. + T*28.33 - T2*0.551 + T3* 0.004)
      B1 = (FS-28.1324)/10.0
      B2 = B1 * B1
      B= -B1* (147.3-T*2.72 + T2*0.04 - D*1.0E-4*(32.4- 0.87*T+.02*T2))
      B= B+ B2*(4.5-0.1*T - D*1.0E-4*(1.8-0.06*T))
      CO = 4886./(1. + 1.83E-5*D)
      ALPHA=     D*1.0E-6* (CO+A+B)
      DN=(SIGMA+ALPHA)/(1.-1.E-3*ALPHA)

	dens(i,j,k,l) = DN
 100	CONTINUE
 2000	CONTINUE

      RETURN
      END
********************************************************************
      REAL FUNCTION POTEM(T,S,P)
C     POTENTIAL TEMPERATURE FUNCTION
C     BASED ON FOFONOFF AND FROESE (1958) AS SHOWN IN "THE SEA" VOL. 1,
C     PAGE 17, TABLE IV
C     INPUT IS TEMPERATURE, SALINITY, PRESSURE (OR DEPTH)
C     UNITS ARE DEG.C., PPT, DBARS (OR METERS)
	REAL t, s, p
	REAL s2, p2, t2, t3, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11
      B1=-1.60E-5*P
      B2=1.014E-5*P*T
      T2=T*T
      T3=T2*T
      B3=-1.27E-7*P*T2
      B4=2.7E-9*P*T3
      B5=1.322E-6*P*S
      B6=-2.62E-8*P*S*T
      S2=S*S
      P2=P*P
      B7=4.1E-9*P*S2
      B8=9.14E-9*P2
      B9=-2.77E-10*P2*T
      B10=9.5E-13*P2*T2
      B11=-1.557E-13*P2*P
      POTEM=B1+B2+B3+B4+B5+B6+B7+B8+B9+B10+B11
      POTEM=T-POTEM
      RETURN
      END
