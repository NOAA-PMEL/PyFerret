	SUBROUTINE  W_ON_TS( uk  , muk,
     .			     vk  , mvk,
     .			     w_t , mw_t )

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
* compute W ( vertical velocity ) on TEMP/SALT columns via zero divergence 
* condition

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/22/87
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
	INTEGER	muk, mvk, mw_t

* subscript ranges from memory variable table ...
	REAL     uk( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		 vk( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		w_t( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, k_lo, k_hi, l_lo, l_hi, i, j, k, l
	REAL	bad_uk, bad_vk, bad_w_t, fact1, fact2, fact3, factor
	REAL	fvn, fvst, fuw(2:imtp1)

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(mw_t)
	i_hi = mr_hi_s1(mw_t)
	j_lo = mr_lo_s2(mw_t)
	j_hi = mr_hi_s2(mw_t)
	k_lo = mr_lo_s3(mw_t)
	k_hi = mr_hi_s3(mw_t)
	l_lo = mr_lo_s4(mw_t)
	l_hi = mr_hi_s4(mw_t)

* flag for bad/missing data
	bad_uk   = mr_bad_data( muk )
	bad_vk	 = mr_bad_data( mvk )
	bad_w_t  = mr_bad_data( mw_t )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
* 2966 C
* 2967 C   FIND ADVECTIVE COEFFICIENTS FOR WEST AND NORTH FACES OF T,S BOXES
* 2968 C
* 2969       FACT1=CSTR(J)*DYTR(J)
* 2970       FACT2=DYTR(J)*CSTR(J)*CS(J)
* 2971       DO 690 K=1,KM
* 2972       DO 690 I=1,IMT
* 2973       FUW(I,K)=(U(I-1,K)*DYU(J)+UM(I-1,K)*DYU(J-1))*FACT1
* 2974       FVN(I,K)=(V(I,K)*DXU2D(I,K)+V(I-1,K)*DXU2D(I-1,K))
* 2975      1 *FACT2*DXT4R2D(I,K)
* 2976  690  CONTINUE
* 2977       IF (NERGY.NE.0) W=0.
* 2978 C
* 2979 C   COMPUTE VERTICAL VELOCITY IN T,S COLUMNS
* 2980 C
* 2981       DO 700 I=1,IMT
* 2982       W(I,1)=0.0
* 2983  700  CONTINUE
* 2984       DO 710 K=1,KM
* 2985       DO 710 I=1,IMT
* 2986       W(I,K+1)=C2DZ2D(I,K)*((FUW(I+1,K)-FUW(I,K))*DXT4R2D(I,K)+FVN(I,K)
* 2987      1    -FVST(I,K))
* 2988  710  CONTINUE
* 2989       DO 712 K=1,KM
* 2990       DO 712 I=1,IMT
* 2991       W(I,K+1)=W(I,K)+W(I,K+1)
* 2992  712  CONTINUE

* 3250 C   SET ADVECTIVE COEFFICIENT FOR NEXT CALL
* 3251 C
* 3252       FACTOR=CST(J)*DYT(J)*CSTR(J+1)*DYTR(J+1)
* 3253       DO 990 K=1,KM
* 3254       DO 990 I=1,IMT
* 3255       FVST(I,K)=FVN(I,K)*FACTOR
* 3256  990  CONTINUE

*******************************************************************
C   COMPUTE VERTICAL VELOCITY IN TEMP/SALT COLUMNS
* south to north loop through east-west lines
	DO 1000 j = j_lo, j_hi

	IF ( j .EQ. 1  .OR.  j .EQ. jm ) THEN
	   DO 100 l = l_lo, l_hi
	   DO 100 k = k_lo, k_hi
	   DO 100 i = i_lo, i_hi
 100	   w_t(i,j,k,l) = bad_w_t
	   GOTO 1000
	ENDIF

C   FIND ADVECTIVE COEFFICIENTS FOR SOUTH AND NORTH FACES OF T,S BOXES	! 2967

      FACT1 = CSTR(j)*DYTR(j)	    				! 2969 mod
      FACT2 = fact1 * CS(j)	 	   			! 2970 mod
      fact3 = fact1 * cs(j-1)

      DO 900 l = l_lo, l_hi
      DO 900 k = k_lo, k_hi

      DO 690 I= MAX(2,i_lo), MIN(imt,i_hi)+1			! 2972 mod
	IF ( uk(i-1,j  ,k,l) .EQ. bad_uk
     .  .OR. uk(i-1,j-1,k,l) .EQ. bad_uk ) THEN
	   fuw(i) = bad_val4
	ELSE
           FUW(I)=(Uk(i-1,j,k,l)*DYU(j)+Uk(I-1,j-1,k,l)*DYU(j-1))*FACT1	! 2973 mod
	ENDIF
 690  CONTINUE	    							! 2976


C   COMPUTE VERTICAL increment to w VELOCITY IN T,S COLUMNS
      DO 710 I = MAX(2,i_lo), MIN(imtm1,i_hi)
	IF ( fuw(i  ) .EQ. bad_val4
     .  .OR. fuw(i+1) .EQ. bad_val4 ) THEN
	   w_t(i,j,k,l) = bad_w_t

	ELSEIF ( vk(i  ,j  ,k,l) .EQ. bad_vk
     .  .OR.     vk(i-1,j  ,k,l) .EQ. bad_vk
     .	.OR.     vk(i-1,j-1,k,l) .EQ. bad_vk ) THEN
	   w_t(i,j,k,l) = bad_w_t

	ELSE
	   FVN  = (Vk(I,j  ,k,l)*DXU(I)+Vk(I-1,j  ,k,l)
     .				*DXU(I-1))*FACT2*DXT4R(I)		! 2975 
	   FVST = (Vk(I,j-1,k,l)*DXU(I)+Vk(I-1,j-1,k,l)
     .				*DXU(I-1))*FACT3*DXT4R(I)
	   w_t(I,j,k,l)=C2DZ(K)*( (FUW(I+1)-FUW(I))*DXT4R(I) + FVN-FVST )
	ENDIF
 710  CONTINUE

* mask out edges ?
	IF ( i_lo .EQ. 1   ) w_t(1  ,j,k,l) = bad_w_t
	IF ( i_hi .EQ. imt ) w_t(imt,j,k,l) = bad_w_t

 900	CONTINUE
 1000	CONTINUE


* now add this dW to w's from above
	DO 255 l = l_lo, l_hi
	DO 255 k = 2   , k_hi
	DO 255 j = j_lo, j_hi
	DO 255 i = i_lo, i_hi

	   IF ( w_t(i,j,k-1,l) .EQ. bad_w_t ) THEN

	      w_t(i,j,k,l) = bad_val4

	   ELSE

	      w_t(i,j,k,l) = w_t(i,j,k-1,l) + w_t(i,j,k,l)

	   ENDIF

 255	CONTINUE

	RETURN
	END
