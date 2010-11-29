	SUBROUTINE GRAD_P_CLINIC   (	rhok   , mrhok   , 
     .					dpdx   , mdpdx   , 
     .					dpdy   , mdpdy 		)

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
* compute horizontal components of grad p assuming constant surface pressure

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/14/87 extracted from SUBROUTINE CLINIC
*			 with major modifications to convert i-k oriented
*			 calculations to i-j oriented calculations
* revision 0.1 - 4/3/87  corrected j offset error
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* V420 - 2/96 - fix bug in bounds checking on j loop

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
	INTEGER	mrhok, mdpdx, mdpdy

* subscript ranges from memory variable table ...
	REAL    rhok( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		dpdx( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		dpdy( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAL	bad_rhok, bad_dpdx, bad_dpdy,
     .		fact1, fact2, udif(im), vdif(im), udif1, vdif1, dpdx1, dpdy1

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(mdpdx)
	i_hi = mr_hi_s1(mdpdx)
	j_lo = mr_lo_s2(mdpdx)
	j_hi = mr_hi_s2(mdpdx)

* flag for bad/missing data
	bad_rhok = mr_bad_data( mrhok )
	bad_dpdx = mr_bad_data( mdpdx )
	bad_dpdy = mr_bad_data( mdpdy )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
*  226 C   FIND HYDROSTATIC PRESSURE GRADIENT AT FIRST LEVEL
*  227 C
*  228       FACT1=GRAV*DZZ(1)*CSR(J)
*  229       FACT2=DYU2R(J)*GRAV*DZZ(1)
*  230       DO 260 I=1,IMT
*  231       UDIF(I,1)=RHON(I+1,1)-RHOS(I,1)
*  232       VDIF(I,1)=RHON(I,1)-RHOS(I+1,1)
*  233       DPDX(I,1)=(FACT1*(UDIF(I,1)-VDIF(I,1)))*DXU2R(I)
*  234       DPDY(I,1)=(UDIF(I,1)+VDIF(I,1))*FACT2
*  235  260  CONTINUE
*  236 C
*  237 C   FIND HYDROSTATIC PRESSURE GRADIENT AT OTHER LEVELS
*  238 C
*  239       FACT1=0.5*CSR(J)*GRAV
*  240       FACT2=DYU4R(J)*GRAV
*  241       DO 270 K=2,KM
*  242       DO 270 I=1,IMT
*  243       DPDX(I,K)=RHON(I,K-1)+RHON(I,K)
*  244       DPDY(I,K)=RHOS(I,K-1)+RHOS(I,K)
*  245  270  CONTINUE
*  246       DO 273 K=2,KM
*  247       DO 273 I=1,IMT
*  248       UDIF(I,K)=DPDX(I+1,K)-DPDY(I  ,K)
*  249       VDIF(I,K)=DPDX(I  ,K)-DPDY(I+1,K)
*  250       DPDX(I,K)=(FACT1*(UDIF(I,K)-VDIF(I,K)))*DZZ2D(I,K)*DXU2R2D(I,K)
*  251       DPDY(I,K)=(FACT2*(UDIF(I,K)+VDIF(I,K)))*DZZ2D(I,K)
*  252  273  CONTINUE
*  253       DO 275 K=1,KMM1
*  254       DO 275 I=1,IMT
*  255       DPDX(I,K+1)=DPDX(I,K)+DPDX(I,K+1)
*  256       DPDY(I,K+1)=DPDY(I,K)+DPDY(I,K+1)
*  257  275  CONTINUE

* ******************************************************************

	DO 2000 l = mr_lo_s4(mdpdx), mr_hi_s4(mdpdx)
	DO 2000 k = mr_lo_s3(mdpdx), mr_hi_s3(mdpdx)

* top level doesn't require recursive computation - deeper levels do
	IF ( k .GT. 1 ) GOTO 500

c find hydrostatic pressure gradient at first level
	DO 260 j = j_lo, j_hi
	   fact1 = grav * dzz(1) * csr(j)
	   fact2 = dyu2r(j) * grav * dzz(1)
	DO 260 i = i_lo, i_hi
* ... cant compute derivatives at fringes of full grid
! pre 2/96	   IF ( i .EQ. im .OR. j .EQ. 1 ) THEN
	   IF ( i .EQ. im .OR. j .EQ. jm ) THEN
	      dpdx(i,j,k,l) = bad_dpdx
	      dpdy(i,j,k,l) = bad_dpdy
	      GOTO 260
	   ENDIF

	   IF ( rhok(i  ,j+1,k,l) .EQ. bad_rhok
     .	    .OR.rhok(i+1,j+1,k,l) .EQ. bad_rhok
     .	    .OR.rhok(i+1,j  ,k,l) .EQ. bad_rhok
     .	    .OR.rhok(i  ,j  ,k,l) .EQ. bad_rhok ) THEN
	      dpdx(i,j,k,l) = bad_dpdx
	      dpdy(i,j,k,l) = bad_dpdy
	   ELSE
	      udif1 = rhok(i+1,j+1,k,l) - rhok(i  ,j  ,k,l)
	      vdif1 = rhok(i  ,j+1,k,l) - rhok(i+1,j  ,k,l)
	      dpdx(i,j,k,l) = ( fact1*(udif1-vdif1) ) * dxu2r(i)
	      dpdy(i,j,k,l) = ( udif1+vdif1 ) * fact2
	   ENDIF
 260	CONTINUE
	GOTO 2000

**************************************************************************
* find hydrostatic pressure gradient at deeper levels
 500	CONTINUE

	DO 270 j = j_lo, j_hi
	   fact1 = 0.5 * csr(j) * grav
	   fact2 = dyu4r(j) * grav

	DO 270 i = i_lo, i_hi
* ... cant compute derivatives at fringes of full grid
! pre 2/96	   IF ( i .EQ. im .OR. j .EQ. 1 ) THEN
	   IF ( i .EQ. im .OR. j .EQ. jm ) THEN
	      dpdx(i,j,k,l) = bad_dpdx
	      dpdy(i,j,k,l) = bad_dpdy
	      GOTO 270
	   ENDIF

	   IF ( rhok(i  ,j+1,k-1,l) .EQ. bad_rhok
     .	   .OR. rhok(i  ,j+1,k  ,l) .EQ. bad_rhok
     .	   .OR. rhok(i  ,j  ,k-1,l) .EQ. bad_rhok
     .	   .OR. rhok(i  ,j  ,k  ,l) .EQ. bad_rhok
     .	   .OR. rhok(i+1,j+1,k-1,l) .EQ. bad_rhok
     .	   .OR. rhok(i+1,j+1,k  ,l) .EQ. bad_rhok
     .	   .OR. rhok(i+1,j  ,k-1,l) .EQ. bad_rhok
     .	   .OR. rhok(i+1,j  ,k  ,l) .EQ. bad_rhok ) THEN
	      dpdx(i,j,k,l) = bad_dpdx
	      dpdy(i,j,k,l) = bad_dpdy
	      GOTO 270
	   ENDIF

* ... compute increment to grad P for this level
* ... combining 2 adjacent rho planes - else results would be on w planes
	   udif1 = ( rhok(i+1,j+1,k-1,l) + rhok(i+1,j+1,k,l) )
     .		 - ( rhok(i  ,j  ,k-1,l) + rhok(i  ,j  ,k,l) )
	   vdif1 = ( rhok(i  ,j+1,k-1,l) + rhok(i  ,j+1,k,l) )
     .		 - ( rhok(i+1,j  ,k-1,l) + rhok(i+1,j  ,k,l) )
	   dpdx1 = ( fact1*( udif1-vdif1 ) ) * dzz(k) * dxu2r(i)
	   dpdy1 = ( fact2*( udif1+vdif1 ) ) * dzz(k)

* ... combine with results from next shallower level
	   dpdx(i,j,k,l) = dpdx(i,j,k-1,l) + dpdx1
	   dpdy(i,j,k,l) = dpdy(i,j,k-1,l) + dpdy1
 270	CONTINUE

 2000	CONTINUE
	RETURN
	END
