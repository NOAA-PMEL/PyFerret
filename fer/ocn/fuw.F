	SUBROUTINE  FUW( psi , mpsi ,
     .			 uavz, muavz,
     .			 u   , mu,
     .			 ufuw, mufuw )

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
* both W_U ( W on uv grid columns ) and horizontal momentum advection

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -1/21/87 extracted from SUBROUTINE CLINIC
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* V200:  7/27/89 - 4D symmetrical version
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
	INTEGER	mpsi, muavz, mu, mufuw

* subscript ranges from memory variable table ...
	REAL     psi( m1lox:m1hix,m1loy:m1hiy,            m1lot:m1hit ),
     .		uavz( m2lox:m2hix,m2loy:m2hiy,            m2lot:m2hit ),
     .		   u( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .		ufuw( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAL	bad_psi, bad_u, bad_uavz,
     .		ufuw_1, sfu_1

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(mufuw)
	i_hi = mr_hi_s1(mufuw)
	j_lo = mr_lo_s2(mufuw)
	j_hi = mr_hi_s2(mufuw)

* flag for bad/missing data
	bad_psi  = mr_bad_data( mpsi  )
	bad_uavz = mr_bad_data( muavz )
	bad_u    = mr_bad_data( mu    )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
*  90  C   FIND ADVECTIVE COEFFICIENT 'FUW' FOR WEST FACE OF U,V BOX
*  91  C
*  92        DATA TAPLEN/5./
*  93        DO 100 I=2,IMT
*  94        SFU(I)=-(P(I,J+1)-P(I,J))*DYUR(J)*AMIN1(HR(I-1,J),HR(I,J))
*  95   100  CONTINUE
*  96        DO 110 K=1,KM
*  97        DO 110 I=1,IMT
*  98        FUW(I,K)=(UCLIN(I,K)+UCLIN(I-1,K))*0.5
*  99   110  CONTINUE
* 100        DO 115 K=1,KM
* 101        DO 115 I=1,IMT
* 102        FUW(I,K)=(FUW(I,K)+SFU(I))*CSR(J)
* 103   115  CONTINUE
*************************************************************************
	DO 2000 l = mr_lo_s4(mufuw), mr_hi_s4(mufuw)
	DO 2000 k = mr_lo_s3(mufuw), mr_hi_s3(mufuw)

C   FIND ADVECTIVE COEFFICIENT 'FUW' FOR WEST FACE OF U,V BOX
	DO 100 j = j_lo, j_hi
	DO 100 i = i_lo, i_hi
* ... cant compute derivatives at fringes of full grid
	   IF ( i .EQ. 1 .OR. j .EQ. jm ) THEN
	      ufuw(i,j,k,l) = bad_val4
	      GOTO 100
	   ENDIF

	   IF (  psi(i  ,j    ,l) .EQ. bad_psi
     .	    .OR. psi(i  ,j+1  ,l) .EQ. bad_psi
     .	    .OR.   u(i  ,j  ,k,l) .EQ. bad_u
     .	    .OR.   u(i-1,j  ,k,l) .EQ. bad_u
     .	    .OR.uavz(i  ,j    ,l) .EQ. bad_uavz
     .	    .OR.uavz(i-1,j    ,l) .EQ. bad_uavz ) THEN

	      ufuw(i,j,k,l) = bad_val4

	   ELSE

	      sfu_1  = -( psi(i,j+1,l)-psi(i,j,l) ) * dyur(j) * hr
	      ufuw_1 =  ( (u(i  ,j,k,l)-uavz(i  ,j,l))
     .			+ (u(i-1,j,k,l)-uavz(i-1,j,l)) ) * 0.5
	      ufuw(i,j,k,l)=( ufuw_1 + sfu_1 ) * csr(j)

	   ENDIF

 100	CONTINUE
 2000	CONTINUE

	RETURN
	END
