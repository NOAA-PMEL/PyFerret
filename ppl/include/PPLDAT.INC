
* V533 *acm*9/01 - changes for dynamic pplus buffer, call PPLLDC, PPLLDX, PPLLDV
*                  via pplld*_envelope.c which check ppl_nsize (new name for 
*                  NSIZE) and allocate more if needed.  Now will pass X rather than
*                  storing in common.
c
c	ppldat
C** 
C**    @(#)PPLDAT.INC	1.1    3/10/88
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
	COMMON/PPLDTA/IBASE, PBUF, PPLMEM_NSIZE
	INTEGER IBASE, PBUF, PPLMEM_NSIZE
c
c       pplmem_nsize  size of plot-data buffer allocated in pplld*envelope.c 
c	ibase	current position in buffer for next data set
c	pbuf	=0 buffer empty
c		=1 buffer not empty
c
c*******************************************************************
