c
c	zgrid
C** 
C**    @(#)ZGRID.INC	1.1    3/10/88
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
	COMMON/ZGRD/IZTYP,IZMRK,IZLEN
c
c	controls the plotting of the positions of ungridded x,y,z data
c
c	iztyp	=0 no points drawn
c		=1 points drawn along the x axis
c		=2 points drawn along the y axis
c		=3 points drawn at each raw input value
c	izmrk	=0 use the default mark
c		=n use mark n to denote the location
c	izlen	number of points in the x,y,z triplets
c
c*******************************************************************
