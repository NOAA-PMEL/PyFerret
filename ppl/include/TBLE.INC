c
c	tble
C** 
C**    @(#)TBLE.INC	1.1    3/10/88
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
      COMMON /TBLE/PTAB(8,4),DTTAB(8,29),FTTAB(8,9),WTTAB(8,11)
      COMMON /TBLE/STTAB(8,19),PCTAB(8,5),UNIT(4,15)
c
      COMMON /NTBLE/NPTAB,NDTTAB,NFTTAB,NWTTAB
      COMMON /NTBLE/NSTTAB,NPCTAB,NUNIT
c
c	contains the data base for BIBO formated headers
c	to be used with subroutine FILL
c
