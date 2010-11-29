	SUBROUTINE  VEL_DIF_XY(	u     , mu     ,
     .				v     , mv     ,
     .				udfh  , mudfh	)

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
* compute horizontal diffusion of velocity components
* note: the identical code applies to both U and V
* ! temp - this is very inefficient code (as is XY_HDIFFUS) that could be
*	greatly sped up by pre-computation of coefficients

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 1/26/87 - extracted from SUBROUTINE CLINIC
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* revision 0.1 7/2/88  - added Am_factor
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include 'ferret.parm'
	include 'gfdl.parm'		! parameter definitions
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xonedim.cmn'	! geometric constants
	include 'xcontext.cmn'
	include 'tmap_dims.parm'	! with maxdsets
	include 'xdset_parms.cmn'	! with am_factor
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'		! parameter definitions
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'	! geometric constants
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'	! with maxdsets
	INCLUDE 'FERRET_CMN:XDSET_PARMS.CMN'	! with Am_factor
#endif

* calling argument declarations:
	INTEGER	mu, mv, mudfh

* subscript ranges from memory variable table ...
	REAL       u( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		   v( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		udfh( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	LOGICAL	HOURLY_DATA
	INTEGER	i_lo, i_hi, j_lo, j_hi, k_lo, k_hi, l_lo, l_hi,
     .		i, j, k, l, dl, lb4
	REAL	bad_u, ccuj, dduj, gguj, hhuj, ew, ns, rest

	REAL	aauj(im), bbuj(im)
	EQUIVALENCE ( aauj(1), aatj(1) ) ,
     .		    ( bbuj(1), bbtj(1) )	! space saving

* --- end of introductory code ---
* use 1 hour separation for exact results on hourly data
	IF ( HOURLY_DATA(mr_grid(mudfh)) ) THEN
	   dl = -1
	ELSE
	   dl = 0
	ENDIF

* limits for calculation
	i_lo = mr_lo_s1(mudfh)
	i_hi = mr_hi_s1(mudfh)
	j_lo = mr_lo_s2(mudfh)
	j_hi = mr_hi_s2(mudfh)
	k_lo = mr_lo_s3(mudfh)
	k_hi = mr_hi_s3(mudfh)
	l_lo = mr_lo_s4(mudfh)
	l_hi = mr_hi_s4(mudfh)

* flag for bad/missing data
	bad_u = mr_bad_data( mu )

* Am coefficient (scaled by 10**7)
	Am = 1.E7 * dp_Am_factor( mr_data_set(mu) )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
* 311 C   ADD IN HORIZONTAL VISCOUS EFFECTS
* 312 C
* 313 C     VARIABLE LATERAL FRICTION
* 314 C
* 315       CCUJ=AM*CST(J+1)*DYTR(J+1)*DYUR(J)*CSR(J)*XJT(J+1)
* 316       DDUJ=AM*CST(J  )*DYTR(J  )*DYUR(J)*CSR(J)*XJT(J)
* 317       DO 322 I=1,IMT
* 318       AAUJ(I)=AM*DXUR(I)*DXTR(I+1)*CSR(J)*CSR(J)*XIT(I+1)*XJV(J)
* 319       BBUJ(I)=AM*DXUR(I)*DXTR(I  )*CSR(J)*CSR(J)*XIT(I)*XJV(J)
* 320       C(I)=CCUJ*XIV(I)
* 321       D(I)=DDUJ*XIV(I)
* 322   322 CONTINUE
* 323 C
* 324       DO 3222 K=1,KM
* 325       DO 3222 I=1,IMT
* 326       ATJ(I,K)=AAUJ(I)
* 327       BTJ(I,K)=BBUJ(I)
* 328       TEMP2D(I,K)=C(I)
* 329 3222  TEMP(I,K)=D(I)
* 330 C
* 331       DO 320 K=1,KM
* 332       DO 320 I=1,IMT
* 333       UA(I,K)=UA(I,K)+ATJ(I,K)*(UB(I+1,K)-UB(I,K))-BTJ(I,K)*
* 334      *(UB(I,K)-UB(I-1,K))
* 335  320  CONTINUE
* 336       DO 323 K=1,KM
* 337       DO 323 I=1,IMT
* 338       UA(I,K)=UA(I,K)+TEMP2D(I,K)*(UBP(I,K)-UB(I,K))-TEMP(I,K)*
* 339      * (UB(I,K)-UBM(I,K))
* 340  323  CONTINUE
* 341       DO 330 K=1,KM
* 342       DO 330 I=1,IMT
* 343       VA(I,K)=VA(I,K)+ATJ(I,K)*(VB(I+1,K)-VB(I,K))-BTJ(I,K)*
* 344      *(VB(I,K)-VB(I-1,K))
* 345  330  CONTINUE
* 346       DO 333 K=1,KM
* 347       DO 333 I=1,IMT
* 348       VA(I,K)=VA(I,K)+TEMP2D(I,K)*(VBP(I,K)-VB(I,K))-TEMP(I,K)*
* 349      * (VB(I,K)-VBM(I,K))
* 350  333  CONTINUE

* 408 C     ALSO ADD IN REST OF VARIABLE HORIZONTAL DIFFUSION
* 409 C
* 410       GGUJ=AM*(1.0-TNG(J)*TNG(J))/(RADIUS*RADIUS)
* 411       HHUJ=2.*AM*SINE(J)/(RADIUS*CS(J)*CS(J))
* 412       DO 3591 K=1,KM
* 413       DO 3591 I=1,IMT
* 414       TEMP(I,K)=GGUJ*XIV(I)*XJV(J)
* 415 3591  TEMP2D(I,K)=HHUJ*XIV(I)*XJV(J)
* 416 C
* 417       DO 360 K=1,KM
* 418       DO 360 I=1,IMT
* 419       UA(I,K)=GM(I,K)*((UA(I,K)-DPDX(I,K)
* 420      * +TEMP(I,K)*U(I,K)-
* 421      *  TEMP2D(I,K)*DXU2R2D(I,K)*(V(I+1,K)-V(I-1,K))))
* 422  360  CONTINUE
* 423       DO 370 K=1,KM
* 424       DO 370 I=1,IMT
* 425       VA(I,K)=GM(I,K)*((VA(I,K)-DPDY(I,K)
* 426      * +TEMP(I,K)*V(I,K)+
* 427      *  TEMP2D(I,K)*DXU2R2D(I,K)*(U(I+1,K)-U(I-1,K))))
* 428  370  CONTINUE
*************************************************************************

	DO 320 j = j_lo, j_hi

* pre-compute some coefficients
	IF ( j .LT. jm ) THEN
	   ccuj=Am*cst(j+1)*dytr(j+1)*dyur(j)*csr(j)*xjt(j+1)
	   dduj=Am*cst(j  )*dytr(j  )*dyur(j)*csr(j)*xjt(j)
	   gguj=Am*(1.0-tng(j)*tng(j))/(radius*radius)
	   hhuj=2.*Am*sine(j)/(radius*cs(j)*cs(j))
	ENDIF
	DO 322 I = i_lo, MIN(i_hi,im-1)
	   aauj(i)=Am*dxur(i)*dxtr(i+1)*csr(j)*csr(j)*xit(i+1)*xjv(j)
	   bbuj(i)=Am*dxur(i)*dxtr(i  )*csr(j)*csr(j)*xit(i)*xjv(j)
 322	CONTINUE

	DO 320 l = l_lo, l_hi
* ... some data may be needed from 1 integration step earlier
	   lb4 = l + dl
	DO 320 k = k_lo, k_hi
	DO 320 i = i_lo, i_hi

	   IF ( i .EQ. 1 .OR. i .EQ. im
     .	  .OR.  j .EQ. 1 .OR. j .EQ. jm ) THEN
	      udfh(i,j,k,l) = bad_val4
	      GOTO 320
	   ENDIF

* V is assumed to have bad points exactly where U does ...
	   IF ( u(i  ,j  ,k,lb4) .EQ. bad_u
     .	   .OR. u(i-1,j  ,k,lb4) .EQ. bad_u
     .	   .OR. u(i+1,j  ,k,lb4) .EQ. bad_u
     .	   .OR. u(i  ,j-1,k,lb4) .EQ. bad_u
     .	   .OR. u(i  ,j+1,k,lb4) .EQ. bad_u ) THEN
	      udfh(i,j,k,l) = bad_val4
	      GOTO 320

	   ELSE
	   ew = ( aauj(i)* (u(i+1,j,k,lb4)-u(i  ,j,k,lb4))
     .		- bbuj(i)* (u(i  ,j,k,lb4)-u(i-1,j,k,lb4)) )
	   ns = ( ccuj   * (u(i,j+1,k,lb4)-u(i,j  ,k,lb4))
     .		- dduj   * (u(i,j  ,k,lb4)-u(i,j-1,k,lb4)) ) * xiv(i)
	   rest = ( gguj*u(i,j,k,l)
     .		- hhuj*dxu2r(i)*(v(i+1,j,k,l)-v(i-1,j,k,l)) )
     .		* xiv(i) * xjv(j)

	   udfh(i,j,k,l) = ( ew + ns + rest ) * cmonthly

	   ENDIF

 320	CONTINUE
	RETURN

	END
