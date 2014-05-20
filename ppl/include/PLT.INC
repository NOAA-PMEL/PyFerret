c
c	plt
C** 
C**    @(#)PLT.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
* v691 5/14 *acm* Change name MOD to TEKMOD
c
	INTEGER TEKMOD
	COMMON/PLT/ ICODE,WIDTH,HEIGHT,TEKMOD,IR,LUN,BATMODE
c
c	icode	plot type
c	width	plot width (inches)
c	height	plot height (inches)
c	tekmod	tek model number
c	ir	baud rate
c	lun	output unit number (unused?)
c	batmode	1=batch (for handling -batch giffile.gif)
c
c*******************************************************************
