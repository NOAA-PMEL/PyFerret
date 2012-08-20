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
c
	COMMON/PLT/ ICODE,WIDTH,HEIGHT,MOD,IR,LUN,BATMODE
c
c	icode	plot type
c	width	plot width (inches)
c	height	plot height (inches)
c	mod	tek model number
c	ir	baud rate
c	lun	output unit number (unused?)
c	batnide	1=batch (for handling -batch giffile.gif)
c
c*******************************************************************
