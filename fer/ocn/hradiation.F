      SUBROUTINE HRADIATION ( sst, msst, rad, mrad )

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
* compute radiative heat flux into sea surface
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/30/86
* revision 1.0 - 4/4/88  - added SST and heat ramp calculations
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

* capitalized code is lifted from the GFDL model
*
*	SW-LW (C5) is short wave down less long wave up

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl.parm'
	include 'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xonedim.cmn'
	include 'ferret_cmn/xdset_parms.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'
	INCLUDE 'FERRET_CMN:XDSET_PARMS.CMN'
#endif

* calling argument declarations:
	INTEGER	msst, mrad
	REAL    sst( m1lox:m1hix,m1loy:m1hiy,            m1lot:m1hit ),
     .		rad( m2lox:m2hix,m2loy:m2hiy,            m2lot:m2hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, l, dset
	REAL	qswd, qlwu, rday, c5, angl, bad_rad, bad_sst, factor

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(mrad)
	i_hi = mr_hi_s1(mrad)
	j_lo = mr_lo_s2(mrad)
	j_hi = mr_hi_s2(mrad)

* flag for bad or missing values
	bad_sst = mr_bad_data ( msst )
	bad_rad = mr_bad_data ( mrad )

* data set needed to determine which calculation to use
	dset = mr_data_set( mrad)

* ... PHILANDER/SIEGEL CODE ...
      QSWD=500.			! cal/cm**2-day
      QLWU=115.			!      "
      RDAY=1./86400.		! 1 / seconds per day
      C5=(QSWD-QLWU)*RDAY

C									!  784
C     APPROX ANNUAL SHORT WAVE DOWN - LONG WAVE UP			!  785
C									!  786
	DO 2000 l = mr_lo_s4(mrad), mr_hi_s4(mrad)

      DO 30 i = i_lo, i_hi
      DO 30 j = j_lo, j_hi					!  787 mod
      ANGL=ACOS(CS(j))*57.29578		! degrees		!  788

* select appropriate calculation for data set
	IF ( dp_phil_qflux( dset ) ) THEN

* ... Philander/Siegel heat flux (doesn't depend on SST)
           rad(i,j,l)=(-ANGL*7.5+650.-QLWU)*RDAY		!  789
           IF(ANGL.LT.20.) rad(i,j,l)=C5			!  790

* ... calculations below depend on SST
	ELSEIF( sst(i,j,l) .EQ. bad_sst ) THEN
	   rad(i,j,l) = bad_rad

	ELSEIF( dp_ramp1_qflux( dset ) ) THEN
* ... single ramp heat flux - hot regions receive less radiation
	   IF ( sst(i,j,l) .GT. 27. ) THEN
	      factor = 1. - ( 82.65 * (sst(i,j,l)-27.) / (3.*500.) )
	   ELSE
	      factor = 1.
	   ENDIF
	   IF ( angl .GT. 20. ) THEN
	      rad(i,j,l) = ( (-angl*7.5+650.)*factor - qlwu ) * rday
	   ELSE
	      rad(i,j,l) = (             qswd*factor - qlwu ) * rday
	   ENDIF

	ELSEIF( dp_ramp2_qflux( dset ) ) THEN
* ... double ramp heat flux - hot regions receive less; cold regions, more
	   IF ( sst(i,j,l) .LT. 26. ) THEN
	      factor = 1. + ( 82.65 * (26.-sst(i,j,l)) / (2.*500.) )
	   ELSEIF ( sst(i,j,l) .GT. 27. ) THEN
	      factor = 1. - ( 82.65 * (sst(i,j,l)-27.) / (3.*500.) )
	   ELSE
	      factor = 1.
	   ENDIF
	   IF ( angl .GT. 20. ) THEN
	      rad(i,j,l) = ( (-angl*7.5+650.)*factor - qlwu ) * rday
	   ELSE
	      rad(i,j,l) = (             qswd*factor - qlwu ) * rday
	   ENDIF

	ELSE
* ... unknown heat flux parameterization
	   rad(i,j,l) = bad_rad
	ENDIF
* compute increment


30    CONTINUE								!  791

 2000	CONTINUE

      RETURN								!  833
      END								!  834
