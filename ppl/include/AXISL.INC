c
c	axisl
C** 
C**    @(#)AXISL.INC	1.1    3/10/88
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
*	Modified time word for whoi format to be 14 digits
*	J Davison 11.19.95
C      *kms* 7/10 changed name of common block from AXISL to AXISLCMN 
C                 to remove clash with AXISL function
* V694  *acm*  1/15 Changes to allow two time axes on plot (F-T)

	COMMON/AXISLCMN/IFRX,IFRY,IT1,IT2,ITSTRT,ITMIN,ITMAX,
     .       IT1b,IT2b,ITSTRTb,ITMINb,ITMAXb

*11.95	CHARACTER IFRX*20,IFRY*20,IT1*10,IT2*10,ITSTRT*10
*11.95	CHARACTER ITMIN*10,ITMAX*10

	CHARACTER IFRX*20,IFRY*20,IT1*14,IT2*14,ITSTRT*14
	CHARACTER ITMIN*14,ITMAX*14
	
	CHARACTER IFRXb*20,IFRYb*20,IT1b*14,IT2b*14,ITSTRTb*14
	CHARACTER ITMINb*14,ITMAXb*14
c
c	ifrx	format for x axis numeric label
c	ifry	format for y axis numeric label
c	it1	minimum time for t axis (YYMMDDHHMMSSCC) !11.95
c	it2	maximum time for t axis (YYMMDDHHMMSSCC)
c	itstrt	time of t=1,  TLO=1.-diff(it1,itstart)/dt (YYMMDDHHMMSSCC)
c	itmin	minimum time for all lines in buffer
c	itmax	maximum time for all lines in buffer
c
c*******************************************************************
