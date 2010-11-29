	SUBROUTINE  VEL_ADV_EW ( fuw , mfuw,
     .				 u   , mu  ,
     .				 uadx, muadx )

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
* compute zonal advection of velocity components
* note: the identical code applies to both U and V

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -1/23/87 extracted from SUBROUTINE CLINIC
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* revision 0.1 - 4/14/87 - converted incorrect flux-form calculation to
*			   true advection
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include 'ferret.parm'
	include 'gfdl.parm'		! parameter definitions
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xonedim.cmn'	! geometric constants
	include 'xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'		! parameter definitions
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'	! geometric constants
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	mfuw, mu, muadx

* subscript ranges from memory variable table ...
	REAL     fuw( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		   u( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		uadx( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAL	bad_u, bad_fuw, dqudx, qdudx

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(muadx)
	i_hi = mr_hi_s1(muadx)
	j_lo = mr_lo_s2(muadx)
	j_hi = mr_hi_s2(muadx)

* flag for bad/missing data
	bad_u   = mr_bad_data( mu )
	bad_fuw = mr_bad_data( mfuw )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
* 282 C   COMPUTE EFFECT OF HORIZONTAL ADVECTION IN MOMENTUM EQUATIONS
* 283 C
* 284       DO 300 K=1,KM
* 285       DO 300 I=1,IMT
* 286       TEMP2D(I,K)=FUW(I+1,K)*(U(I+1,K)+U(I,K))
* 287  300  CONTINUE
* 288       DO 303 K=1,KM
* 289       DO 303 I=1,IMT
* 290       UA(I,K)=(TEMP2D(I-1,K)-TEMP2D(I,K))*DXU2R2D(I,K)
* 291  303  CONTINUE

* 297       DO 310 K=1,KM
* 298       DO 310 I=1,IMT
* 299       TEMP2D(I,K)=FUW(I+1,K)*(V(I+1,K)+V(I,K))
* 300  310  CONTINUE
* 301       DO 313 K=1,KM
* 302       DO 313 I=1,IMT
* 303       VA(I,K)=(TEMP2D(I-1,K)-TEMP2D(I,K))*DXU2R2D(I,K)
* 304  313  CONTINUE
*************************************************************************
	DO 2000 l = mr_lo_s4(muadx), mr_hi_s4(muadx)
	DO 2000 k = mr_lo_s3(muadx), mr_hi_s3(muadx)

	DO 300 j = j_lo, j_hi

* to save half of the additions ...
*	   IF ( i .NE. 1 ) tempim1 = fuw(i,j,k,l) * (u(i,j,k,l)+u(i-1,j,k,l))

	DO 300 i = i_lo, i_hi

	   IF ( i .EQ. 1 .OR. i .EQ. im ) THEN
	      uadx(i,j,k,l) = bad_val4
	      GOTO 300
	   ENDIF

	   IF ( u(i-1,j,k,l) .EQ. bad_u
     .	    .OR.u(i  ,j,k,l) .EQ. bad_u
     .	    .OR.u(i+1,j,k,l) .EQ. bad_u
     .	    .OR.fuw(i  ,j,k,l) .EQ. bad_fuw
     .	    .OR.fuw(i+1,j,k,l) .EQ. bad_fuw ) THEN
  
	      uadx(i,j,k,l) = bad_val4

	   ELSE

! temp - future performance:  tempi = fuw(i+1,j,k,l) * (u(i+1,j,k,l)+u(i,j,k,l))

* flux-form calculation: compute +d/dx(QU) ( where "Q" may be "U" or "V" )
	      dqudx = ( fuw(i+1,j,k,l) * (u(i+1,j,k,l)+u(i,j,k,l)) 
     .		    -   fuw(i  ,j,k,l) * (u(i,j,k,l)+u(i-1,j,k,l)) ) * dxu2r(i)

* compute +Q*d/dx(U)
	      qdudx = u(i,j,k,l) * (fuw(i+1,j,k,l) - fuw(i,j,k,l)) * dxur(i)

* compute -U*d/dx(Q) = Q*d/dx(U) - d/dx(QU)
	      uadx(i,j,k,l) = ( qdudx - dqudx ) * cmonthly

	   ENDIF

 300	CONTINUE
 2000	CONTINUE
	RETURN
	END
