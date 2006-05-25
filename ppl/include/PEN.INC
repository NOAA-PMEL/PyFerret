c
c	pen
C** 
C**    @(#)PEN.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
* V600 *acm* 5/06  num_line_colors for MODE LINECOLORS

c
	COMMON/PEN/ IPEN(0:NLINES),IP(4),num_line_colors
c
c	ipen(0:)pen to use for each line, ipen(0) is default
c	ip(1:4)	misc flags
c	ip(1)	=0 no box around plot
c		=1 draw box
c	ip(2)	=0 do not echo command input to echo file
c		=1 echo on
c	ip(3)	=0 interactive
c		=1 batch
c       num_line_colors = number of line colors allowed (see ws_fill_rep.F)
c
c*******************************************************************
