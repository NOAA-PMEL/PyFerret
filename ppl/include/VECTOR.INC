c
c	vector
C** 
C**    @(#)VECTOR.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
	COMMON/VECTOR/VLEN,VUSRLN,VAUTO,VXLOC,VYLOC,VUSER,VKEY,
     *	VPOS,VFRMT,VSKPX,VSKPY,VMINLN,ARMAX,ARMIN,HFACT
	LOGICAL VAUTO,VKEY,VUSER
	INTEGER VSKPX,VSKPY
	CHARACTER VFRMT*20
c
c	vlen	vector length (inches) used for key length
c	vusrln	user unit value for vlen vector
c	vauto	logical if true auto scale
c	vxloc	x location for key
c	vyloc	y location for key
c	vuser	logical if true user units for position
c	vkey	logical if true draw the key
c	vpos	justification for key
c	vfrmt	format to use for key label
c	vskpx	skip x
c	vskpy	skip y
c	vminln	minimum vector size
c	armax	size for maximum head size
c	armin	size for minimum head size
c	hfact	size factor for head
c
c*******************************************************************
