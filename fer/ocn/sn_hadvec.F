	SUBROUTINE SN_HADVEC ( tk, mtk, vk, mvk, advec, madvec )

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
* revision 0.0 - 3/17/86	extracted from SUBROUTINE HDIFFUS
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* revision 0.1 - 6/24/88 - added bad data check on vk(i,j-1)
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - array declarations using XMEM_SUBSC.CMN (reordered args)

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
	INTEGER	mvk, mtk, madvec
* subscript ranges from memory variable table ...
	REAL       tk( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		   vk( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		advec( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )
	REAL	fvst(im),fvn(im)

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAl	bad_tk, bad_vk, bad_qadvec, fact1, fact2, fact3, dvtdym, tdvdy

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(madvec)
	i_hi = mr_hi_s1(madvec)
	j_lo = mr_lo_s2(madvec)
	j_hi = mr_hi_s2(madvec)

* flag for bad/missing data
	bad_tk	   = mr_bad_data( mtk )
	bad_vk	   = mr_bad_data( mvk )
	bad_qadvec = mr_bad_data( madvec )

* south to north loop through east-west lines
	DO 2000 l = mr_lo_s4(madvec), mr_hi_s4(madvec)
	DO 2000 k = mr_lo_s3(madvec), mr_hi_s3(madvec)

	DO 1000 j = j_lo, j_hi

	IF ( j .EQ. 1  .OR.  j .EQ. jm ) THEN
	   DO 100 i = i_lo, i_hi
 100	   advec( i,j,k,l ) = bad_qadvec
	   GOTO 1000
	ENDIF

C   FIND ADVECTIVE COEFFICIENTS FOR SOUTH AND NORTH FACES OF T,S BOXES	! 2967

      FACT1 = CSTR(j)*DYTR(j)	    				! 2969 mod
      FACT2 = fact1 * CS(j)	 	   			! 2970 mod
      fact3 = fact1 * cs(j-1)
      DO 690 I= MAX(2,i_lo), i_hi				! 2972 mod
	IF ( vk(i  ,j  ,k,l) .EQ. bad_vk
     .  .OR. vk(i-1,j  ,k,l) .EQ. bad_vk
     .  .OR. vk(i  ,j-1,k,l) .EQ. bad_vk
     .	.OR. vk(i-1,j-1,k,l) .EQ. bad_vk ) THEN
	   fvn(i) = bad_val4
	ELSE
	   FVN (I)=(Vk(i,j,k,l  )*DXU(I)+Vk(I-1,j  ,k,l)
     .		   *DXU(I-1))*FACT2*DXT4R(I)			! 2975 mod
	   FVST(I)=(Vk(i,j-1,k,l)*DXU(I)+Vk(I-1,j-1,k,l)
     .		   *DXU(I-1))*FACT3*DXT4R(I)
	ENDIF
 690  CONTINUE	    							! 2976

* calculate minus south-north component of total advection in flux form
      DO 820 I= i_lo, i_hi					! 3066 mod

	
	IF ( i .EQ. 1
     .	.OR. fvn(i    ) .EQ. bad_val4
     .	.OR. tk (i,j  ,k,l) .EQ. bad_tk
     .	.OR. tk (i,j-1,k,l) .EQ. bad_tk
     .	.OR. tk (i,j+1,k,l) .EQ. bad_tk ) THEN
	   advec( i,j,k,l ) = bad_qadvec
	   GOTO 820
	ENDIF

        dvtdym =  (						! 3069 mod
     *           -FVN (I)*(tk(i,j+1,k,l) + tk(i,j,  k,l))	! 3070 mod
     *           +FVST(I)*(tk(i,j,k,l  ) + tk(i,j-1,k,l)))	! 3071 mod

* calculate individual advection component by subtracting the vector t*dU/dX 
	tdvdy = tk(i,j,k,l) * (fvn(i  ) - fvst(i))         *2.0
	advec(i,j,k,l) = (dvtdym + tdvdy ) * cmonthly

 820  CONTINUE	    							! 3072

 1000	CONTINUE
 2000	CONTINUE

	RETURN
	END
