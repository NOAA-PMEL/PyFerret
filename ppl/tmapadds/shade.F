	SUBROUTINE SHADE (z)

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
*	Modified of PLOTZ to get data into proper grids for 
*	SHADE command

*	J Davison 8.26.88
*
*	Mod JD 4.18.90 to allow data all of the same value to be shaded. 
*       MOD *JD* To permit use of centered levels and specify the delta Z
*       value as well.  Added variables to CONT.INC common 8.3.95  
*	   *jd 8.28 fix bug in dz specification
*	*jd* 2.96 Change centered option so that data not spanning zero
*			is centered as well.
*	*jd* 5.96 Fix bug where if shade/lev=N option is used with constant 
*	field, no shade results
*       *jd* 1.98 Fix bug: shade misses top level, when top level is lt 0.

#ifdef unix
	include		'parampl5_dat.decl'
	include		'PARAMPL5.DAT'
#else
	INCLUDE		'tmap_pplv11inc:tmap_parampl5.dat'
#endif

#ifdef unix
	include		'axis_inc.decl'
	include		'AXIS.INC'
	include		'plt_inc.decl'
	include		'PLT.INC'
	include		'miss_inc.decl'
	include		'MISS.INC'
	include		'hd_inc.decl'
	include		'HD.INC'
	include		'cont_inc.decl'
	include		'CONT.INC'
	include		'zgrid_inc.decl'
	include		'ZGRID.INC'
	include		'zgridd_inc.decl'
	include		'ZGRIDD.INC'
#else
	INCLUDE		'tmap_pplv11inc:tmap_axis.inc'
	INCLUDE		'tmap_pplv11inc:tmap_plt.inc'
	INCLUDE		'tmap_pplv11inc:tmap_miss.inc'
	INCLUDE		'tmap_pplv11inc:tmap_hd.inc'
	INCLUDE		'tmap_pplv11inc:tmap_cont.inc'
	INCLUDE		'tmap_pplv11inc:tmap_zgrid.inc'
	INCLUDE		'tmap_pplv11inc:tmap_zgridd.inc'
#endif

	real		dx,dy,dz,z(nx,ny),znp,zmp

	integer		i,j
	
        parameter       (epsilon = 1e-6)

C       Add external stmt for linux port *jd* 12.96 RANGE is intrinsic fcn
        external range

******************************************************************************

C       
C       IAUTOC = 1 means: compute levels from input data and number
C                         of levels

	IF (IAUTOC.EQ.1) THEN

	   IF (ZMIN.EQ.ZMAX) THEN

C             NLEV2=0 is the code in pplmod_plotz
	      NLEV2=1

	   ELSE
C       
C       If set_delta is .true. the delta value has been set
C       elsewhere and we should calculate the max and min values
C       of the range based on this delta.
C       (This portion stolen from RANGE)

	      if (set_delta) then 

		 dz = delta
C       
C       CALCULATE THE MINIMUM VALUE OF THE RANGE
C       
		 FM = zmin / DZ
		 M = IFIX(FM)
		 IF (FM .LT. 0.0) M = M - 1
		 znp = DZ * FLOAT(M)
C       
C       CALCULATE THE MAXIMUM VALUE OF THE RANGE.
C       
		 FM = zmax / DZ
		 M = IFIX(FM)
		 IF(FM .GT. 0.0)M = M + 1
		 zmp = DZ * FLOAT(M)

C       Else we should allow RANGE to determine the max and min range
C       values and also the appropriate delta value. (ZNP, ZMP, DZ)

	      else

		 CALL RANGE(ZMIN,ZMAX,NLEV,ZNP,ZMP,DZ)

	      endif

C       Set levels centered about zero if centered is .true.   
 
	      if (centered) then
		 if (abs(znp) .gt. abs(zmp)) then
		    zmp = -znp
		 else
		    znp = -zmp
		 endif
	      endif

	      NLEV2=(ZMP-ZNP)/DZ+1 

C       Handle rounding errors *jd* 6.27.95/12.30.97

	      IF (ZMP .GT. 0.0) THEN
		 IF (ZNP + DZ*(NLEV2-1) .LT. ZMP) 
	1	      NLEV2 = NLEV2 + 1
	      ELSE
		 IF (ABS(ZNP + DZ*(NLEV2-1) - ZMP) .LE. 
	1	     ABS(EPSILON*ZMP)) 
	1	      NLEV2 = NLEV2 + 1
	      ENDIF

	      DO 100 I=1,NLEV2 
		 ZLEV(I)=ZNP+DZ*FLOAT(I-1) 
 100	      CONTINUE 

	   ENDIF

C       
C       Else IAUTOC != 1 means: use supplied levels
C       
	ELSE 

C       Mod 4.18.90 to allow just on level to be SHADEd
C       CONTOUR requires more than one level to work but SHADE doesn't

	   NLEV2=NLEV

	END IF 


C       DRAW
C
C       from ppl/include/PPL.INC:
C
C       JTYPE: = 2 2-d irregular rectangular grid
C              = 1 1-d set
C              = 0 2-d regular rectangular grid (integer)
C              =-1 2-d second component of a vector regular grid
C              =-2 2-d second component of a vector irregular grid
C

C
C       If it's a 2-d irregular grid, the irregular grid values
C       are also stored in the Z array.  Pass them to shade_sub.

	IF (JTYPE.EQ.2 .OR. JTYPE.EQ.-2) THEN

	   CALL SHADE_SUB(Z,Z(1,NY+1),Z(1,NY+2),XX(IZSE-NX-1),YY(IZSE-NY-1))

C       For all other plot types fill in XX() and YY() by
C       multiplying the index value by the DX or DY delta values.

	ELSE

	   DX = (XMAX - XMIN)/(NX - 1)
	   DY = (YMAX - YMIN)/(NY - 1)

	   DO 500 I=1,NX
	      XX(IZLEN+I)=DX*(I-1)+XMIN
 500	   CONTINUE
	   DO 510 J=1,NY
	      YY(IZLEN+J)=DY*(J-1)+YMIN
 510	   CONTINUE

c *kob* 8/97 had to move the conintation marker to col 5 - it was in col 8
c       this bombed aix compiler, though it seems to work on others
	   CALL SHADE_SUB(Z,XX(IZLEN+1),YY(IZLEN+1),XX(IZLEN+NX+1),
     1	       YY(IZLEN+NY+1))

	ENDIF

	RETURN
	END




