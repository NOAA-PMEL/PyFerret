      SUBROUTINE HFLUX ( vel, mvel, sst, msst, air, mair,
     .			 sens, msens,  evap, mevap )

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
* compute heat flux into sea surface
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 2.0 - 3/7/86	major modifications from GFDLCONT Rev 5.4 to convert
*			i-k oriented calculations to i-j oriented calculations
* revision 2.1 -4/4/88 - air temp is pre-computed for various parameterizations
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - array declarations using XMEM_SUBSC.CMN (reordered args)

* capitalized code is lifted from the GFDL model
*
* equations used:
*	Q = SW - LW - QS - QE
* where
*	Q is heat flux
*	SW-LW (C5) is short wave down less long wave up
*	QS is sensible heating: = ro*Cd*Cp*V*(Tsst-Tair)
*	QE is evaporative cooling: = ro*L*Cd*V*(qradi(Tsst) - R*qradi(Tair) )
*	where
*		qs(T) = 0.622es(T)/PA and
*		qe(T) = 6.11*exp[L/Rv * (1/273 - 1/T) ]

* note: in the original GFDL version of this subroutine the surface Q flux was
*	imbedded in the array TDIF with suitable constants so that the surface
*	flux appeared to be diffusing from level k=0 of the ocean.  Several of
*	the variables went under non-explicit EQUIVALENCE names so that the
*	effective calculation (where "qflux" is the surface heat flux) became:
* 		tdif(i,1)= sst + dzz(1)/bvdc(i) * qflux(i)	- in HFLUX
*		delta_T  = bvdc(i)*eeh(1) * (tdif(i,1)-sst(1))	- in TRACER 3106
*	where	eeh(1)   = 1/(dz(1)*dzz(1))

#ifdef unix
	include 'ferret.parm'
	include 'gfdl.parm'
	include 'tmap_dims.parm'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xonedim.cmn'	! geometric constants
	include 'xcontext.cmn'
	include 'xdset_parms.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'	! geometric constants
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XDSET_PARMS.CMN'
#endif

* calling argument declarations:
	INTEGER	mair, mvel, msst, msens, mevap
	REAL	 vel( m1lox:m1hix,m1loy:m1hiy,            m1lot:m1hit ),
     .		 sst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		 air( m3lox:m3hix,m3loy:m3hiy,            m3lot:m3hit ),
     .		sens( m4lox:m4hix,m4loy:m4hiy,            m4lot:m4hit ),
     .		evap( m5lox:m5hix,m5loy:m5hiy,            m5lot:m5hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, l, dset
	REAL	bad_air, bad_sst, bad_vel, bad_sens, bad_evap, vel1,
     .		d1, d2, windmin
	REAL	R, RO, EL, CD, PA, CP, C1, C2, rap

* --- end of introductory code ---
* initialize
	dset    = mr_data_set( msens )
	windmin = dp_min_wind( dset )

* limits for calculation
	i_lo = mr_lo_s1(msens)
	i_hi = mr_hi_s1(msens)
	j_lo = mr_lo_s2(msens)
	j_hi = mr_hi_s2(msens)

* flag for bad/missing data
	bad_air  = mr_bad_data( mair )
	bad_vel  = mr_bad_data( mvel )
	bad_sst  = mr_bad_data( msst )
	bad_sens = mr_bad_data( msens )
	bad_evap = mr_bad_data( mevap )

C     SET CONSTANTS							!  769
C									!  770
      R=.8
      RO=1.2E-3			! (of air) gm/cm**3
      EL=595.			! cal/gm
      CD=1.4E-3
      PA=1013.25		! millibars
      CP=.24			! cal/gm-deg
      C1=RO*CD*EL*.662/PA
      C2=RO*CD*CP
	rap = 1. / (ro * cd)	! translated from RAP assignment below

	DO 2000 l = mr_lo_s4(msens), mr_hi_s4(msens)

	DO 300 j = j_lo, j_hi
	DO 300 i = i_lo, i_hi

* check air, SST and wind stress
	IF ( air(i,j  ,l) .EQ. bad_air
     .	.OR. sst(i,j,1,l) .EQ. bad_sst
     .	.OR. vel(i,j  ,l) .EQ. bad_vel ) GOTO 290

C     CALC WIND MAGNITUDE FROM STRESS					!  803
C(     DON'T ALLOW WIND MAGNITUDE < 488 CM/SEC (.40DYNES/CM**2)		!  811)
C   don't allow wind magnitude < windmin
C     TO APPROX HIGH FREQ DISTURBANCES IN ITCZ				!  812
      vel1=RAP*VEL(i,j,l)					!  808 mod
      vel1=SQRT(vel1)						!  809 mod
      IF (vel1.LT.windmin) vel1=windmin				!  815 mod
50    CONTINUE							!  816
C								!  817
C     SATURATION VAPOR PRESSURES ... LATENT HEATING		!  818
C								!  819
      d1   =9.4051-2353./( air(i,j  ,l) + 273.16 )		!  820 mod
      d2   =9.4051-2353./( SST(i,j,1,l) + 273.16 )		!  821 mod
      d2   =10.**d2						!  822 mod
      d1   =10.**d1						!  823 mod
      d1   =d2-R*d1						!  824 mod
      d1   =C1*d1						!  825 mod
	sens(i,j,l) =-c2 * vel1 * (sst(i,j,1,l)-air(i,j,l))     ! sensible heat
	evap(i,j,l) =-d1 * vel1				! evaporative heat
	GOTO 300
	
* cannot do calculation at this point
 290	      sens(i,j,l) = bad_sens
	      evap(i,j,l) = bad_evap

 300	continue

 2000	CONTINUE

	RETURN
	END
