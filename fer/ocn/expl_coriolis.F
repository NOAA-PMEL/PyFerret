	SUBROUTINE EXPL_CORIOLIS( u   , mu   , v   , mv   ,
     .				  ucoe, mucoe, vcoe, mvcoe    )

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
* compute the explicit Coriolis effect based on velocity
* components from the preceeding time step
*			correction based on an implicit scheme

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -1/27/87 extracted from SUBROUTINE CLINIC
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
	INTEGER	 mu, mv, mucoe, mvcoe

* subscript ranges from memory variable table ...
	REAL       u( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		   v( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		ucoe( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .		vcoe( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit )

* internal variable declarations:
	LOGICAL	HOURLY_DATA
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l, dl
	REAL	bad_u, factor

* --- end of introductory code ---

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
* 388 C   ADD IN CORIOLIS FORCE
* 389 C
* 390       FACTOR=2.0*OMEGA*SINE(J)
* 391       IF(ACOR.EQ.0) GO TO 358
* 392       DO 357 K=1,KM
* 393       DO 357 I=1,IMT
* 394       UA(I,K)=UA(I,K)+FACTOR*VB(I,K)
* 395       VA(I,K)=VA(I,K)-FACTOR*UB(I,K)
* 396  357  CONTINUE
* 397       GO TO 3590
* 398  358  CONTINUE
* 399       DO 359 K=1,KM
* 400       DO 359 I=1,IMT
* 401       UA(I,K)=UA(I,K)+FACTOR*V(I,K)
* 402       VA(I,K)=VA(I,K)-FACTOR*U(I,K)
* 403  359  CONTINUE
* 404  3590 CONTINUE

*************************************************************************
* compute the explicit Coriolis increment implied by the past velocity

* use 1 hour separation for exact results on hourly data
	IF ( HOURLY_DATA(mr_grid(mucoe)) ) THEN
	   dl = 1
	ELSE
	   dl = 0
	ENDIF

* limits for calculation
	i_lo = mr_lo_s1(mucoe)
	i_hi = mr_hi_s1(mucoe)
	j_lo = mr_lo_s2(mucoe)
	j_hi = mr_hi_s2(mucoe)

* flag for bad/missing data
	bad_u = mr_bad_data( mu )

* compute:
	DO 200 j = j_lo, j_hi

	   factor = 2.0*omega*sine(j) * cmonthly
*	   IF ( acor.EQ.0 ) GOTO 358

	   DO 200 l = mr_lo_s4(mucoe), mr_hi_s4(mucoe)
	   DO 200 k = mr_lo_s3(mucoe), mr_hi_s3(mucoe)

* based on velocity at time n-1:
	   DO 100 i = i_lo, i_hi 
	      IF ( u(i,j,k,l) .EQ. bad_u ) THEN
	         ucoe(i,j,k,l) = bad_val4
	         vcoe(i,j,k,l) = bad_val4
	      ELSE
	         ucoe(i,j,k,l) =  factor * v(i,j,k,l)
	         vcoe(i,j,k,l) = -factor * u(i,j,k,l)
	      ENDIF
 100	   CONTINUE
	   GOTO 200

* based on velocity at time n:
! 358 	   CONTINUE
!	   STOP 'Coriolis based on single time step not implemented'
*	   DO 110 i = i_lo, i_hi 
*	      IF ( u(i,j,k,l+dl) .EQ. bad_u ) THEN
*	         ucoe(i,j,k,l) = bad_val4
*	         vcoe(i,j,k,l) = bad_val4
*	      ELSE
*	         ucoe(i,j,k,l) =  factor * v(i,j,k,l+dl)
*	         vcoe(i,j,k,l) = -factor * v(i,j,k,l+dl)
*	      ENDIF
* 110	   CONTINUE

 200	CONTINUE
	RETURN

	END
