c
c	comepv
C** 
C**    @(#)COMEPV.INC	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
	common /comepv/ iepv1,iepv2,ieflg
c
c**
c
c	iepv1	EPIC generic variable code (integer) for x variable
c	iepv2	EPIC generic variable code (integer) for y-variable
c	ieflg	=0 Data format not 'EPIC' (Cleared with VARS)
c		=1 Data format is 'EPIC' (iform(1:3)='EPI') (Set with
c                  EVAR)
c
c***********************************************************************
