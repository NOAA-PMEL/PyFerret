	SUBROUTINE U_FROM_PSI (	psi , mpsi ,
     .				uavz, muavz,
     .				vavz, mvavz )

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
* compute vertically averaged components of velocity from transport stream fcn

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -1/20/87 extracted from SUBROUTINE CLINIC
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
	INTEGER	mpsi, muavz, mvavz

* subscript ranges from memory variable table ...
	REAL     psi( m1lox:m1hix,m1loy:m1hiy,            m1lot:m1hit ),
     .		uavz( m2lox:m2hix,m2loy:m2hiy,            m2lot:m2hit ),
     .		vavz( m3lox:m3hix,m3loy:m3hiy,            m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, l
	REAL	bad_psi, bad_uavz, bad_vavz,
     .		diag1, diag2

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(muavz)
	i_hi = mr_hi_s1(muavz)
	j_lo = mr_lo_s2(muavz)
	j_hi = mr_hi_s2(muavz)

* flag for bad/missing data
	bad_psi  = mr_bad_data( mpsi  )
	bad_uavz = mr_bad_data( muavz )
	bad_vavz = mr_bad_data( mvavz )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
*  129       C   COMPUTE EXTERNAL MODE VELOCITIES FOR ROW J+1 AND TWO TIME LEVELS
*  137             DO 155 I=1,IMTM1
*  138             DIAG1=P (I+1,J+2)-P (I,J+1)
*  139             DIAG2=P (I,J+2)-P (I+1,J+1)
*  140             SFU (I)=-(DIAG1+DIAG2)*DYU2R(J+1)*HR(I,J+1)
*  141             SFV(I)=(DIAG1-DIAG2)*DXU2R(I)*HR(I,J+1)*CSR(J+1)
*  142        155  CONTINUE
* ******************************************************************
	DO 2000 l = mr_lo_s4(muavz), mr_hi_s4(muavz)

	DO 155 j = j_lo, j_hi
	DO 155 i = i_lo, i_hi
* ... cant compute derivatives at fringes of full grid
	   IF ( i .EQ. im .OR. j .EQ. jm ) THEN
	      uavz(i,j,l) = bad_uavz
	      vavz(i,j,l) = bad_vavz
	      GOTO 155
	   ENDIF

	   IF ( psi(i  ,j  ,l) .EQ. bad_psi
     .	    .OR.psi(i+1,j  ,l) .EQ. bad_psi
     .	    .OR.psi(i+1,j+1,l) .EQ. bad_psi
     .	    .OR.psi(i  ,j+1,l) .EQ. bad_psi) THEN
	      uavz(i,j,l) = bad_uavz
	      vavz(i,j,l) = bad_vavz
	   ELSE
	      diag1=psi(i+1,j+1,l) - psi(i  ,j,l)
	      diag2=psi(i  ,j+1,l) - psi(i+1,j,l)
	      uavz(i,j,l) = -(diag1+diag2) * dyu2r(j) * hr
	      vavz(i,j,l) =  (diag1-diag2) * dxu2r(i) * hr * csr(j)
	   ENDIF
 155	CONTINUE

 2000	CONTINUE
	RETURN
	END
