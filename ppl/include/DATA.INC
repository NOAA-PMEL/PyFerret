c
c	data
C** 
C**    @(#)DATA.INC	1.1    3/10/88
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
	COMMON/DATA/ ITYPEZ,JX,JY,JZ,LUNN,LINEN,NREC,NVAR, 
     *	NGRP,CAY,NRNG 
c
c	itypez	=0 read data by rows (1st subscript varies fastest)
c		=1 read data by columns (2nd varies fastest)
c	jx	position of x variable in data file
c	jy	position of y variable in data file
c	jz	position of z variable in data file
c	lunn	logical unit to read data on
c	linen	number of data lines in buffer
c	nrec	number of data records to read
c	nvar	number of variables per group
c	ngrp	number of groups per record
c	cay	ZGRID interpolation method
c	nrng	number of grid points to interpolate without data ZGRID
c
c*******************************************************************
