	SUBROUTINE  W_ON_UV( ufuw, mufuw,
     .			     vfvn, mvfvn,
     .			     w_u , mw_u )

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
	INTEGER	mufuw, mvfvn, mw_u

* subscript ranges from memory variable table ...
	REAL    ufuw( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		vfvn( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		 w_u( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAL	bad_ufuw, bad_vfvn, bad_w_u, factor

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(mw_u)
	i_hi = mr_hi_s1(mw_u)
	j_lo = mr_lo_s2(mw_u)
	j_hi = mr_hi_s2(mw_u)

* flag for bad/missing data
	bad_ufuw = mr_bad_data( mufuw )
	bad_vfvn = mr_bad_data( mvfvn )
	bad_w_u  = mr_bad_data( mw_u )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
* 700  C   TRANSFER THINGS IN PREPARATION FOR NEXT CALL
* 701  C
* 702        FACTOR=CS(J)*DYU(J)*CSR(J+1)*DYUR(J+1)
* 703        DO 644 K=1,KM
* 704        DO 644 I=1,IMT
* 705        FVSU(I,K)=FVN(I,K)*FACTOR
* 706   644  CONTINUE

* 214  C   COMPUTE VERTICAL VELOCITY IN U,V COLUMNS
* 215  C
* 216        DO 250 K=1,KM
* 217        DO 250 I=1,IMT
* 218        W(I,K+1)=C2DZ2D(I,K)*((FUW(I+1,K)-FUW(I,K))*DXU2R2D(I,K)+FVN(I,K)
* 219       1    -FVSU(I,K))
* 220   250  CONTINUE
* 221        DO 255 K=1,KM
* 222        DO 255 I=1,IMT
* 223        W(I,K+1)=W(I,K)+W(I,K+1)
* 224   255  CONTINUE
*************************************************************************
	DO 2000 l = mr_lo_s4(mw_u), mr_hi_s4(mw_u)
	DO 2000 k = mr_lo_s3(mw_u), mr_hi_s3(mw_u)

C   COMPUTE VERTICAL VELOCITY IN U,V COLUMNS
* first compute incremental dW for this layer
	DO 250 j = j_lo, j_hi
	   IF ( j .GT. 1 ) factor = cs(j-1) * dyu(j-1) * csr(j) * dyur(j)
	DO 250 i = i_lo, i_hi

	   IF ( j .EQ. 1 .OR. i .EQ. im ) THEN
	      w_u(i,j,k,l) = bad_val4
	      GOTO 250
	   ENDIF

	   IF ( ufuw(i  ,j  ,k,l) .EQ. bad_ufuw
     .	    .OR.ufuw(i+1,j  ,k,l) .EQ. bad_ufuw
     .	    .OR.vfvn(i  ,j  ,k,l) .EQ. bad_vfvn
     .	    .OR.vfvn(i  ,j-1,k,l) .EQ. bad_vfvn ) THEN

	      w_u(i,j,k,l) = bad_val4

	   ELSE

	      w_u(i,j,k,l) = c2dz(k)*
     .			( (ufuw(i+1,j,k,l)-ufuw(i,j,k,l)) * dxu2r(i) 
     .			 + vfvn(i,j,k,l) - (factor * vfvn(i,j-1,k,l)) )

	   ENDIF

 250	CONTINUE

* now add this dW to w's from above
	IF ( k .GT. 1 ) THEN
	   DO 255 j = j_lo, j_hi
	   DO 255 i = i_lo, i_hi

	      IF ( w_u(i,j,k-1,l) .EQ. bad_w_u ) THEN

	         w_u(i,j,k,l) = bad_val4

	      ELSE

	         w_u(i,j,k,l) = w_u(i,j,k-1,l) + w_u(i,j,k,l)

	      ENDIF

 255	   CONTINUE

	ENDIF

 2000	CONTINUE

	RETURN
	END
