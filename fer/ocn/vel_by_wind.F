	SUBROUTINE  VEL_BY_WIND(  tau, mtau,
     .				 ubwi, mubwi )

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
* compute contribution to velocity in surface layer due to wind stress
* note1: the same code applies to both X and Y components
* note2: a value of rho=1 is assumed for dimensional correctness

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -1/23/87 extracted from SUBROUTINE CLINIC
*			with major modifications to bypass th fudging of wind
*			stress effects into the diffusion terms
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include 'ferret.parm'
	include 'gfdl.parm'		! parameter definitions
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
#endif

* calling argument declarations:
	INTEGER	mtau, mubwi

* subscript ranges from memory variable table ...
	REAL     tau( m1lox:m1hix,m1loy:m1hiy,            m1lot:m1hit ),
     .		ubwi( m2lox:m2hix,m2loy:m2hiy,            m2lot:m2hit )

* internal variable declarations:
	INTEGER	MGRID_SIZE, grid_size
	REAL	factor

* --- end of introductory code ---

* total words in result grid
	grid_size = MGRID_SIZE( mubwi )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
*259  C   SET VERTICAL VISCOUS BOUNDARY CONDITIONS
*260  C
*261        DO 280 K=1,KM
*262        DO 280 I=1,IMT
*263        UDIF(I,K)=UB(I,K)
*264        VDIF(I,K)=VB(I,K)
*265   280  CONTINUE
*266        TAPER=1.
*267        IF(ITT*DTUV.LE.TAPLEN*86400.) TAPER=0.5*(1.-COS(PI*
*268       1(ITT*DTUV/(TAPLEN*86400.))))
*269        TFAC1=(MOD(ITT-1,NTS30)+1.)/FLOAT(NTS30)
*270        TFAC2=1.-TFAC1
*271        DO 290 I=1,IMT
*272        FACTOR=(DZZ(1)/BVVC(I))*TAPER
*273        UOVER(I)=UB(I,1)+(WSX(I)*TFAC2+TX(I)*TFAC1)*FACTOR
*274        VOVER(I)=VB(I,1)+(WSY(I)*TFAC2+TY(I)*TFAC1)*FACTOR
*275   290  CONTINUE

*362        DO 345 K=1,KMP1
*363        DO 345 I=1,IMT
*364        TEMP2D(I,K)=UDIF(I,K-1)-UDIF(I,K)
*365   345  CONTINUE
*366        DO 348 K=1,KM
*367        DO 348 I=1,IMT
*368        UA(I,K)=UA(I,K)+EEM2D(I,K)*VVC(I,K-1)*TEMP2D(I,K)-
*369       * FFM2D(I,K)*VVC(I,K)*TEMP2D(I,K+1)
*370   348  CONTINUE
	
*378        DO 355 K=1,KMP1
*379        DO 355 I=1,IMT
*380        TEMP2D(I,K)=VDIF(I,K-1)-VDIF(I,K)
*381   355  CONTINUE
*382        DO 356 K=1,KM
*383        DO 356 I=1,IMT
*384        VA(I,K)=VA(I,K)+EEM2D(I,K)*VVC(I,K-1)*TEMP2D(I,K)-
*385       * FFM2D(I,K)*VVC(I,K)*TEMP2D(I,K+1)
*386   356  CONTINUE
*************************************************************************

* when all the fudging is done the entire contribution of wind stress reduces
* to TAUi / dz(1)
* factor also converts to units of cm/sec per month (730 hours )

	factor = ( 1. / dz(1) ) * cmonthly

	CALL COPY_GRID( tau, mtau, ubwi, mubwi )

	CALL MULT_GRID( grid_size, ubwi, factor )

	RETURN

	END
