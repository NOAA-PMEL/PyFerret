	SUBROUTINE  VEL_BY_GRAD_P( dpdx, mdpdx,
     .				   ubp, mubp )

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
* essentially this routine performs only a change of units from sec-1 to month-1
* note1: the same code applies to both X and Y components
* note2: a value of rho=1 is assumed for dimensional correctness

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -1/26/87 
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
	INTEGER	mdpdx, mubp

* subscript ranges from memory variable table ...
	REAL    dpdx( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		 ubp( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	INTEGER	MGRID_SIZE, grid_size
	REAL	factor

* --- end of introductory code ---

* total words in result grid
	grid_size = MGRID_SIZE( mubp )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
* 406 C   ADD IN PRESSURE TERM AND ZERO OUT LAND

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
* in the 205 code the effectd of DPDX are simple subtracted from UA

* change the sign from how it is used in the 205 code so that all of the
* incremental pieces get added together to form the velocity increment
	factor = -1. * cmonthly

	CALL COPY_GRID( dpdx, mdpdx, ubp, mubp )

	CALL MULT_GRID( grid_size, ubp, factor )

	RETURN

	END
