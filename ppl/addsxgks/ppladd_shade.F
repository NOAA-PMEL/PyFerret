#ifndef unix
	OPTIONS /CHECK=NOBOUNDS
#endif
	SUBROUTINE SHADE (z)

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

#ifdef unix
	include		'tmap_pplv11inc/parampl5_dat.decl'
	include		'pplv11inc/PARAMPL5.DAT'
#else
	INCLUDE		'tmap_pplv11inc:tmap_parampl5.dat'
#endif

#ifdef unix
	include		'tmap_pplv11inc/axis_inc.decl'
	include		'pplv11inc/AXIS.INC'
	include		'tmap_pplv11inc/plt_inc.decl'
	include		'pplv11inc/PLT.INC'
	include		'tmap_pplv11inc/miss_inc.decl'
	include		'pplv11inc/MISS.INC'
	include		'tmap_pplv11inc/hd_inc.decl'
	include		'pplv11inc/HD.INC'
	include		'tmap_pplv11inc/cont_inc.decl'
	include		'pplv11inc/CONT.INC'
	include		'tmap_pplv11inc/zgrid_inc.decl'
	include		'pplv11inc/ZGRID.INC'
	include		'tmap_pplv11inc/zgridd_inc.decl'
	include		'pplv11inc/ZGRIDD.INC'
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

C       Handle rounding errors *jd* 6.27.95
C       JC_TODO This is where the rounding error for negative plots probably exists

	      IF (ZNP + DZ*(NLEV2-1) .LT. ZMP) NLEV2 = NLEV2 + 1

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

	   CALL SHADE_SUB(Z,XX(IZLEN+1),YY(IZLEN+1),XX(IZLEN+NX+1),
	1	YY(IZLEN+NY+1))

	ENDIF

	RETURN
	END
