	SUBROUTINE  VEL_ADV_NS ( fvn , mfvn,
     .				 v   , mv  ,
     .				 vady, mvady )

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
	INTEGER	mfvn, mv, mvady

* subscript ranges from memory variable table ...
	REAL     fvn( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		   v( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		vady( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAL	bad_v, bad_fvn, factor, dQVdy, QdVdy, fvst_ij

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(mvady)
	i_hi = mr_hi_s1(mvady)
	j_lo = mr_lo_s2(mvady)
	j_hi = mr_hi_s2(mvady)

* flag for bad/missing data
	bad_v   = mr_bad_data( mv )
	bad_fvn = mr_bad_data( mfvn )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
* 282       C   COMPUTE EFFECT OF HORIZONTAL ADVECTION IN MOMENTUM EQUATIONS

* 292       DO 305 K=1,KM
* 293       DO 305 I=1,IMT
* 294       UA(I,K)=UA(I,K)
* 295      2 -FVN(I,K)*(UP(I,K)+U(I,K))+FVSU(I,K)*(U(I,K)+UM(I,K))
* 296  305  CONTINUE

* 305       DO 315 K=1,KM
* 306       DO 315 I=1,IMT
* 307       VA(I,K)=VA(I,K)
* 308      2 -FVN(I,K)*(VP(I,K)+V(I,K))+FVSU(I,K)*(V(I,K)+VM(I,K))
* 309  315  CONTINUE
*************************************************************************

	DO 305 j = j_lo, j_hi

	   IF ( j .GT. 1 ) factor = cs(j-1) * dyu(j-1) * csr(j) * dyur(j)

	DO 305 l = mr_lo_s4(mvady), mr_hi_s4(mvady)
	DO 305 k = mr_lo_s3(mvady), mr_hi_s3(mvady)
	DO 305 i = i_lo, i_hi

	   IF ( j .EQ. 1 .OR. j .EQ. jm ) THEN
	      vady(i,j,k,l) = bad_val4
	      GOTO 305
	   ENDIF

	   IF ( v(i,j-1,k,l) .EQ. bad_v
     .	    .OR.v(i,j  ,k,l) .EQ. bad_v
     .	    .OR.v(i,j+1,k,l) .EQ. bad_v
     .	    .OR.fvn(i,j  ,k,l) .EQ. bad_fvn
     .	    .OR.fvn(i,j-1,k,l) .EQ. bad_fvn ) THEN
  
	      vady(i,j,k,l) = bad_val4

	   ELSE

* flux-form calculation ( note 1/dy imbedded in FVN )
* "Q" may be "U" or "V"
	      fvst_ij = fvn(i,j-1,k,l) * factor
	      dQVdy   =( fvn(i,j,k,l) * (v(i,j+1,k,l)+v(i,j,k,l  ))
     .		       - fvst_ij      * (v(i,j,k,l  )+v(i,j-1,k,l)) )

* compute Q*d/dy(V)
	      QdVdy = v(i,j,k,l) * ( fvn(i,j,k,l) - fvst_ij ) * 2.0

* compute -Vd/dy(Q) = Qd/dy(V) - d/dy(QV)
	      vady(i,j,k,l) = ( QdVdy - dQVdy ) * cmonthly

	   ENDIF

 305	CONTINUE
	RETURN
	END
