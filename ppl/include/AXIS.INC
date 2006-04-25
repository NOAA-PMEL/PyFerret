c
c	axis
C** 
C**    @(#)AXIS.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c v552 *acm* 3/03 new param IAXSET to restore axes after PLOT/AXES/set plot.
c v554 *acm* 2/04 add /HGRAT /VGRAT to draw optional graticule lines at tic marks
c V600 *acm* 2/06 add PPL XFOR('DMS') or 'DM' sets axis format deg/min/sec or deg/min
c                 add xvaloff, yvaloff (see comments below)
c
	COMMON/AXIS/ XLO,XHI,XTIC,XLEN,XCSIZE, 
     *	IAUTOX,ITYPEX,NMTCX,LINTX,LABELX,NSIGX,NTICX,XORG,XVALOFF,
     *		     YLO,YHI,YTIC,YLEN,YCSIZE, 
     *	IAUTOY,ITYPEY,NMTCY,LINTY,LABELY,NSIGY,NTICY,YORG,YVALOFF,
     *	IAXON(4),IAXSET, HGRAT(4,2), VGRAT(4,2), HLINE, VLINE, HCOLOR(2), 
     *  VCOLOR(2),XDMS,YDMS
c
c**	x axis variables
c
c	xlo	minimum value -- left end  (user units)
c	xhi	maximum value -- right end (user units)
c	xtic	distance between large tics  (user units)
c	xlen	length of axis  (inches)
c	xcsize	height of numeric labels  (inches)
c	iautox	=0 use supplied xlo,xhi,xtic
c		=1 calculate xlo,xhi,xtic from data and nticx
c	itypex	=1 linear axis top and bottom
c		=2 normal log axis top and bottom
c		=3 normal log axis bottom, inverse log axis top 
c	nmtcx	number of small tics between each large tic
c	lintx	label interval of large tics, i.e. 2 every other tic
c	labelx	=-1 label the bottom axis
c		=0  suppress the labels
c		=1  label the top axis
c	nsigx	number of significant digits if ifrx = ' '
c	nticx	number of large tics to try for if iautox=1
c	xorg	position of left end of axis from left boundry (inches)
c	xvaloff	add this amount to each x axis label. (x axis data was read as
c               double precision and converted to single after applying an offset.)
c	
c**	y axis variables
c
c	ylo	minimum value -- bottom end  (user units)
c	yhi	maximum value -- top end (user units)
c	ytic	distance between large tics  (user units)
c	ylen	length of axis  (inches)
c	ycsize	height of numeric labels  (inches)
c	iautoy	=0 use supplied ylo,yhi,ytic
c		=1 calculate ylo,yhi,ytic from data and nticy
c	itypey	=1 linear axis top and bottom
c		=2 normal log axis top and bottom
c		=3 normal log axis bottom, inverse log axis top 
c	nmtcy	number of small tics between each large tic
c	linty	label interval of large tics, i.e. 2 every other tic
c	labely	=-1 label the left axis
c		=0  suppress the labels
c		=1  label the right axis
c	nsigy	number of significant digits if ifry = ' '
c	nticy	number of large tics to try for if iautoy=1
c	yorg	position of bottom end of axis from bottom boundry (inches)
c	yvaloff	add this amount to each y axis label. (y axis data was read as
c               double precision and converted to single after applying an offset.)
c
c**	axis flags
c       iaxset          =1 if axes have been reset (needed if plot/set/axes=,
c                       so we can reset after the ppl plot)
c	iaxon(1:4)	four flags controlling the plotting of each axis
c			=0 suppress plotting of axis and character label
c			=1 plot axis and character label
c	iaxon(1)	top axis (x axis)
c	iaxon(2)	bottom axis (x axis)
c	iaxon(3)	left axis (y axis)
c	iaxon(4)	right axis (y axis)
c       HGRAT      draw horizontal tic lines across whole plot 0=no,1=dash,2=solid
c       VGRAT      draw vertical tic lines across whole plot 0=no,1=dash,2=solid
c       hline           flag: true if still need to draw horizontal tic lines
c       vline           flag: true if still need to draw vertical tic lines
c       XDMS           = 0 for degrees in decimal deg.frac
c                      = 1 for Degrees and minutes 
c       YDMS           same as xdms, for vertical axis
c
c*******************************************************************
