c
c	zgridd
C** 
C**    @(#)ZGRIDD.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C 9/2013 *ACM*	See ticket 2092. These arrays are the coordinate 
C               data for 2D graphics. Increase the size to match
C               Ferret's storage limit on irregular coordinates.
C
	PARAMETER (IZSE=750000)
	COMMON/ZGRIDD/XX(IZSE),YY(IZSE),ZZ(IZSE)
c
c	xx()	x variable as read in
c	yy()	y variable as read in
c	zz()	z variable as read in
c*******************************************************************
