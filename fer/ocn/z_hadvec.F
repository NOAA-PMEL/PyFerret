	SUBROUTINE Z_HADVEC ( tk, mtk,
     .			      wk, mwk,
     .			      advec, madvec   )

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
* compute vertical  component of heat advection

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/17/86	extracted from SUBROUTINE HDIFFUS
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - array declarations using XMEM_SUBSC.CMN (reordered args)
* Linux Port 1/97 *kob* - Added a preprocessor include for xgfdl_masks.cmn
*			  because it needed a preprocessor.
* V450 - 7/22/97 - minor reordering of "#includes" for Solaris compiler bug

#ifdef unix
	include 'ferret.parm'
	include 'gfdl.parm'		! parameter definitions

! land/water masks
#	include "xgfdl_masks.cmn"	

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
	INCLUDE 'FERRET_CMN:XGFDL_MASKS.CMN'	! land/water masks
#endif

* calling argument declarations:
	INTEGER	mwk, mtk, madvec
* subscript ranges from memory variable table ...
	REAL       tk( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		   wk( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		advec( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAL	bad_wk, bad_tk, bad_advec, up_dwtdz, dn_dwtdz, dwtdz_neg,
     .		tdwdz
	LOGICAL	  bad_components

* --- end of introductory code ---
* limits for calculation
	i_lo = mr_lo_s1(madvec)
	i_hi = mr_hi_s1(madvec)
	j_lo = mr_lo_s2(madvec)
	j_hi = mr_hi_s2(madvec)

* flag(s) for bad or missing values
	bad_wk    = mr_bad_data ( mwk )
	bad_tk    = mr_bad_data ( mtk )
	bad_advec = mr_bad_data ( madvec )

	DO 2000 l = mr_lo_s4(madvec), mr_hi_s4(madvec)
	DO 2000 k = mr_lo_s3(madvec), mr_hi_s3(madvec)

* south to north loop through east-west lines
	DO 1000 j = j_lo, j_hi

	   IF ( j .LE. 2  .OR.  j .GE. jmtm1 ) THEN
	      DO 100 i = i_lo, i_hi
 100	      advec( i,j,k,l ) = bad_advec
	      GOTO 1000
	   ENDIF

* calculate minus vert advection component of total advection in flux form
* note on sign: increasing k is decreasing z
* note on k levels: The surface (w=0) is at k=1 in GFDL code but is
*		    omitted from the GFDL output. In this code k=1 represents
*		    the first level of non-zero data (GFDL k=2).
*		    The k=1 level of the S-T and U-V grids is therefore ABOVE
*		    the k=1 level of W in this code.  The "up" and "dn" prefixes
*		    on the variable names should be thought of relative to the
*		    T level which is experiencing advective heat flux.

      DO 824 i = i_lo, i_hi

* check that all of the required component fields are available at this point
* depending on depth, some of the components may be dummies
	IF     ( k .EQ. 1 )  THEN
	   bad_components =	wk( i,j,k  ,l ) .EQ. bad_wk
     .			   .OR. tk( i,j,k  ,l ) .EQ. bad_tk
     .			   .OR. tk( i,j,k+1,l ) .EQ. bad_tk
	ELSEIF ( k .EQ. KM ) THEN
	   bad_components =	wk( i,j,k-1,l ) .EQ. bad_wk
     .			   .OR. wk( i,j,k  ,l ) .EQ. bad_wk
     .			   .OR. tk( i,j,k-1,l ) .EQ. bad_tk
     .			   .OR. tk( i,j,k  ,l ) .EQ. bad_tk
	ELSE
	   bad_components =	wk( i,j,k-1,l ) .EQ. bad_wk
     .			   .OR. wk( i,j,k  ,l ) .EQ. bad_wk
     .			   .OR. tk( i,j,k-1,l ) .EQ. bad_tk
     .			   .OR. tk( i,j,k  ,l ) .EQ. bad_tk
     .			   .OR. tk( i,j,k+1,l ) .EQ. bad_tk
	ENDIF
	IF ( bad_components ) THEN
	   advec( i,j,k,l ) = bad_advec
	   GOTO 824
	ENDIF

	IF ( k .EQ. 1 ) THEN
	   up_dwtdz = 0.0
	ELSE
	   up_dwtdz = -(wk(i,j,k-1,l) * (tk(i,j,k-1,l) + tk(i,j,k,l)) )
     .		    * DZ2R(K)					    ! 3075 mod
	ENDIF

	IF ( k .GE. kmt(i,j) ) THEN
	   dn_dwtdz = 0.0
	ELSE
	   dn_dwtdz = (wk(i,j,k,l)*(tk(i,j,k,l)+tk(i,j,k+1,l)))*DZ2R(K)! 3079 mod
	ENDIF

	dwtdz_neg = up_dwtdz + dn_dwtdz

* calculate actual vert advection by subtracting t*dW/dZ ...
	IF ( k .EQ. 1 ) THEN
	   tdwdz = tk(i,j,k,l) * (              - wk(i,j,k,l)) * dz2r(k) * 2.0
	ELSE
	   tdwdz = tk(i,j,k,l) * (wk(i,j,k-1,l) - wk(i,j,k,l)) * dz2r(k) * 2.0
	ENDIF
	advec(i,j,k,l) = ( dwtdz_neg + tdwdz ) * cmonthly

 824  CONTINUE	    							! 3080

 1000	CONTINUE
 2000	CONTINUE

	RETURN
	END
