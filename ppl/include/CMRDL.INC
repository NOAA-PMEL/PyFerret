c
c	cmrdl
C** 
C**    @(#)CMRDL.INC	1.1    3/10/88
C**
C** v5.51+ *acm* 3/03 Longer strings: changes for multi-line labels.
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
	COMMON/CMRDL/ ICOM,LABEL,IFORM,STRING
	CHARACTER ICOM*30,LABEL*2048,IFORM*80,STRING*2048
c
c	icom	command parsed by COMRD5
c	label	character portion of line parsed
c	iform	data format examples are:
c		DSF
c		BIBO
c		EPIC
c		UNF
c		FREE
c		(F5.0)
c	string	input line used by GETCMD
c
c*******************************************************************
