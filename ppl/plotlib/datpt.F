	subroutine datapt(xlow,ylow,xhigh,yhigh,xlplot,ylplot, 
     1	xhplot,yhplot,xp,yp,zp,npts,imark,forml,hgtlab) 
C** 
C**    @(#)datpt.f	1.1    3/10/88
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
c     this routine posts data points xp(k),yp(k),zp(k),k=1,npts 
c     on a plot.  complot routines size,scale,tktype etc. should 
c     have already been called. 
c 
c     xlow,ylow:   the lower left corner of the plot in user units 
c     xhigh,yhigh: upper right corner in user units 
c     xlplot,ylplot:lower left corner of plot in inches. 
c     xhplot,yhplot:upper right corner of plot in inches. 
c     xp:          array containing x coordinates of data pts. 
c     yp:          array containing y coordinates. 
c     zp:          array containing z coordinates (this value is posted) 
c     npts:        number of data points to be posted. 
c     imark:       mark type. see complot documentation. 
c     forml:       format to be used for the data pt label. 
c                  example: (f10.3). label can be up to 30 chars long. 
c     hgtlab:      height of the label in inches. use 0.0 for no label. 
c 
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	dimension xp(npts),yp(npts),zp(npts)
	character label*81,forml*(*) 
 
c     statement functions: 
	xloc(a)=(a-xlow)*xfact+xlplot 
	yloc(a)=(a-ylow)*yfact+ylplot 
 
	xfact=(xhplot-xlplot)/(xhigh-xlow) 
	yfact=(yhplot-ylplot)/(yhigh-ylow) 
 
	do 500 jpt=1,npts 
	xinch=xloc(xp(jpt)) 
	yinch=yloc(yp(jpt)) 
 
	if(imark.le.0)go to 100 
	call plot(xinch,yinch,0,0) 
	call mark(imark) 
 
100	if(hgtlab.le.0.0)go to 500 
	read(label,forml)zp(jpt)
	nchar=lnblk(label,81)
	width=symwid(hgtlab,nchar,label) 
 
	ybias=0.08 
c        if no mark is specified then center the label 
c        on the data point: 
	if(imark.le.0)ybias=-0.5*hgtlab 
	if(yinch+ybias.le.ylplot)ybias=ylplot-yinch+0.08 
	if(yinch+ybias+hgtlab.ge.yhplot)ybias=-hgtlab-0.08 
 
	xbias=-0.5*width 
	if(xinch+xbias.le.xlplot)xbias=xlplot-xinch+0.08 
	if(xinch+xbias+width.ge.xhplot)xbias=xhplot-xinch-width-0.08 
 
	call symbel(xinch+xbias,yinch+ybias,0.0,hgtlab,nchar,label) 
 
500	continue 
 
	return 
	end 
