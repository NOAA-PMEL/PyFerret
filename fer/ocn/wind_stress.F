	SUBROUTINE WIND_STRESS ( uwind,  muwind,
     .				 vwind,	 mvwind,
     .				 stress, mstress )

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
* compute wind stress magnitude on T-S grid from components on U-V grid
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/17/86	extracted from SUBROUTINE HDIFFUS
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* revision 0.1 - 1/27/89 - modified bad_flag error checking at j=2 in case of
*			   improper GFDL model output
*		note: bad flag checks are based on UWND, only for performance
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* Unix/RISC port 8/91 *sh*: added dp_nts30_wind sensitivity
*                         - also fixed bug at j=2 and j=jmtm1 by changing to
*                           ELSEIF logic
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI

#ifdef unix
	include 'ferret.parm'
	include 'gfdl.parm'
        include 'tmap_dims.parm'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'
        include 'xdset_parms.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
        INCLUDE 'FERRET_CMN:XDSET_PARMS.CMN'
#endif

* calling argument declarations:
	INTEGER	muwind, mvwind, mstress
* subscript ranges from memory variable table ...
	REAL     uwind( m1lox:m1hix,m1loy:m1hiy,            m1lot:m1hit ),
     .		 vwind( m2lox:m2hix,m2loy:m2hiy,            m2lot:m2hit ),
     .		stress( m3lox:m3hix,m3loy:m3hiy,            m3lot:m3hit )

* internal variable declarations:
        LOGICAL need_nts30
	INTEGER	dset, grid, i_lo, i_hi, j_lo, j_hi, i, j, l,
     .          itt, ittmod, slen
	REAL	bad_wind, bad_stress, taux, tauy
        REAL*8  TM_WORLD
	CHARACTER LEFINT*6

* local parameters
        INTEGER nts30
        PARAMETER ( nts30 = 730 )

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(mstress)
	i_hi = mr_hi_s1(mstress)
	j_lo = mr_lo_s2(mstress)
	j_hi = mr_hi_s2(mstress)

* flag(s) for bad or missing values
	bad_wind   = mr_bad_data ( muwind )
	bad_stress = mr_bad_data ( mstress )

* data set needed to determine whether to do NST30 modification
        dset = mr_data_set( muwind )
        grid = mr_grid    ( muwind )

C									! 3003
C     SINCE STRESS & AIRT ARE DEFINED CORRECTLY ONLY FOR THE CURRENT	! 3007
C     JROW DURING THE 2 TIME STEPS WHEN BC ARE BROUGHT IN AT THE	! 3008
C     BEGINNING OF EACH MONTH...APPROXIMATE THEM ON JROW-1 BY USING	! 3009
C     THE JROW VALUES							! 3010
C									! 3011
* note:	Ron Pacanowski agrees that this IF test is incorrect and...
*      IF (MOD(ITT-1,NTS30).EQ.0.OR.MOD(ITT-1,NTS30).EQ.1) THEN		! 3012
*	should be replaced by:
!	if (j.eq.2) then
****** this will slightly alter the results ******
* note 8/91: this incorrect test remained in throughout the life of
* the 205 model and critically effects the balances when attempting
* an exact match using hourly data.
* The descriptor parameter "NTS30 WIND STRESS MOD" was created to turn back on
* this feature
C									! 3013
	DO 2000 l = mr_lo_s4(mstress), mr_hi_s4(mstress)

* special correction at nts30 time steps ?
* NOTE: nts30 logic is only partially tested - month starts, only tested
*       also, there is a bug (8/91) in the heat flux calculation at present
*       when using hourly data:  qflx is entirely calculated at T-1.  In
*       fact sst should be at T-1 but tau should be at T.  If that bug is
*       fixed the MOD logic above will need to be changed
        IF ( dp_nts30_wind( dset ) ) THEN
	   CALL SPLIT_LIST(pttmode_bull, 6,
     .			'WIND_STRESS: Checking itt against nts30',0)
           itt = INT( TM_WORLD( l, grid, t_dim, box_middle ) )
           ittmod = MOD(itt+1,nts30)
           need_nts30 = ittmod.EQ.0 .OR. ittmod.EQ.1
           IF ( need_nts30 ) 	   CALL SPLIT_LIST(pttmode_bull, 6,
     .			'Using NTS30 at '//LEFINT(l,slen)//
     .			LEFINT(itt,slen),0)
        ELSE
           need_nts30 = .FALSE.
        ENDIF

* shift winds from u,v grid to T grid points
* via 4-point interpolation where possible
	DO 1720 j = j_lo, j_hi
	DO 1720 I = i_lo, i_hi					! 3021 mod

	   IF ( j .EQ. 1  .OR.  i .EQ. 1 ) THEN
	      stress(i,j,l) = bad_stress

	   ELSEIF (      uwind(i  ,j,l  ) .EQ. bad_wind
     .		.OR. uwind(i-1,j,l  ) .EQ. bad_wind ) THEN
	      stress(i,j,l) = bad_stress

	   ELSEIF ( j.EQ.2 .OR. j.EQ.jmtm1 .OR. need_nts30 )  THEN ! 2 pt intrp
	      TAUX = .5*( uwind(i,j,l) + uwind(i-1,j,l) )	! 3015 mod
	      TAUY = .5*( vwind(i,j,l) + vwind(i-1,j,l) )	! 3017 mod
	      stress(i,j,l) = SQRT ( taux**2 + tauy**2 )

	   ELSEIF (  uwind(i  ,j-1,l) .EQ. bad_wind
     .		.OR. uwind(i-1,j-1,l) .EQ. bad_wind ) THEN
	      stress(i,j,l) = bad_stress

           ELSE
	      TAUX = .25* ( uwind (i,j  ,l) + uwind (i-1,j  ,l) +    ! 3022 mod
     .			    uwind (i,j-1,l) + uwind (i-1,j-1,l) )    ! 3023 mod
	      TAUY = .25* ( vwind (i,j  ,l) + vwind (i-1,j  ,l) +    ! 3024 mod
     .			    vwind (i,j-1,l) + vwind (i-1,j-1,l) )    ! 3025 mod
	      stress(i,j,l) = SQRT ( taux**2 + tauy**2 )
	   ENDIF

 1720	CONTINUE								! 3026
 2000	CONTINUE								! 3026

	RETURN
	END
