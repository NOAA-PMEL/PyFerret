	SUBROUTINE Z_HDIFFUS (	tk,   mtk,
     .				vdc,  mvdc,
     .				qflx, mqflx,
     .				qdif, mqdif )

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
* compute vertical component of heat diffussion using temperature at
* 3 layers (tkm,tk,tkp) and vertical diffussion coefficient at 2 (vdcM,vdc)
* and using surface boundary fluxes if at the surface

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/7/86	extracted from SUBROUTINE HDIFFUS
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* revision 0.1 - 6/25/87 - modified to include surface heat flux when k=1
* V200:  7/25/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
*	10/18/89 - fixed bug: lb4-->l on setting qdif to bad_flag while adding
*				      surface BC's
* Linux Port 1/97 *kob* - Added a preprocessor include for xgfdl_masks.cmn
*			  because it needed a preprocessor.
* V450 - 7/22/97 - minor reordering of "#includes" for Solaris compiler bug

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl.parm'		! parameter definitions

! land/water masks
#	include "ferret_cmn/xgfdl_masks.cmn"	

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
	INCLUDE 'FERRET_CMN:XGFDL_MASKS.CMN'	! land/water masks
#endif

* calling argument declarations:
	INTEGER	mtk, mvdc, mqflx, mqdif

* subscript ranges from memory variable table ...
	REAL      tk( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		 vdc( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		qflx( m3lox:m3hix,m3loy:m3hiy,            m3lot:m3hit ),
     .		qdif( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit )

* internal variable declarations:
	LOGICAL HOURLY_DATA, need_qflux
	INTEGER	i, j, k, l, dl, lb4
	REAL	bad_vdc, bad_tk, bad_qdiffus, bad_qflx, up_qdif, dn_qdif

* --- end of introductory code ---
* use 1 hour separation for exact results on hourly data
	IF ( HOURLY_DATA(mr_grid(mqdif)) ) THEN
	   dl = -1
	ELSE
	   dl = 0
	ENDIF

* flag(s) for bad or missing values
	bad_vdc      = mr_bad_data ( mvdc )
	bad_tk       = mr_bad_data ( mtk  )
	bad_qflx     = mr_bad_data ( mqflx )
	bad_qdiffus  = mr_bad_data ( mqdif )

* flag for surface heat flux
	DO 1100 l = mr_lo_s4(mqdif), mr_hi_s4(mqdif)
	DO 1100 k = mr_lo_s3(mqdif), mr_hi_s3(mqdif)

* components from 1 hour before ?
	lb4 = l + dl

	need_qflux = k .EQ. 1

	DO 1000 j = mr_lo_s2(mqdif), mr_hi_s2(mqdif)
	DO 1000 i = mr_lo_s1(mqdif), mr_hi_s1(mqdif)

* diffussion from above:
	IF     ( k .EQ. 1        ) THEN
	   up_qdif = 0.0			! top
	ELSEIF ( k .GT. kmt(i,j) ) THEN
	   up_qdif = 0.0			! bottom or below
	ELSE	

	   IF (vdc( i,j,k-1,lb4 ) .EQ. bad_vdc
     .	   .OR.	tk( i,j,k-1,lb4 ) .EQ. bad_tk
     .	   .OR. tk  ( i,j,k,lb4 ) .EQ. bad_tk ) THEN
	      qdif( i,j,k,l ) = bad_qdiffus
	      GOTO 1000
	   ENDIF

	   up_qdif = cmonthly * (
     *		 vdc(i,j,k-1,lb4)*EEH(K)
     .				*(tk(i,j,k-1,lb4) - tk(i,j,k,lb4)))! 3105 mod
	ENDIF

* diffussion to below:
	IF ( k .GE. kmt(i,j) ) THEN
	   dn_qdif = 0.0			! bottom or below
	ELSE

	   IF (	vdc( i,j,k,  lb4 ) .EQ. bad_vdc
     .	   .OR. tk ( i,j,k+1,lb4 ) .EQ. bad_tk ) THEN
	      qdif( i,j,k,l ) = bad_qdiffus
	      GOTO 1000
	   ENDIF

	   dn_qdif = cmonthly * (
     *		-VDC (i,j,k,lb4)*FFH(K)*(tk(i,j,k,lb4)-tk(i,j,k+1,lb4))) ! 3106 mod
	ENDIF

	qdif(i,j,k,l) = up_qdif + dn_qdif

* add in surface boundary flux if at surface
* (with units converted from cgs to deg/month over upper layer)
	IF ( need_qflux ) THEN
	   IF ( qflx(i,j,lb4) .EQ. bad_qflx ) THEN
	      qdif(i,j,k,l) = bad_qdiffus
	   ELSE
	      qdif(i,j,k,l) = qdif(i,j,k,l)
     .				 + 0.06265833*qflx(i,j,lb4)
	   ENDIF
	ENDIF

 1000	CONTINUE
 1100	CONTINUE

	RETURN
	END
