c
c	cont
C** 
C**    @(#)CONT.INC	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
* v581  *acm* 3/05 open levels (-INF) (INF)
* V610  *acm* 3/08 - add new MODE NLEVELS
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels
* V65 *acm   2/10 If Open-ended levels were used, symbol LEV_OPNLEVS saves 
* the /LEV command needed to duplicate the levels


	COMMON/CONT/ZLEV(ICNLEV),LDIG(ICNLEV),LWGT(ICNLEV),LCLR(ICNLEV),
     *	NLEV,NLEV_DFLT,HGT,LEV_STD,LEV_MEAN,LEV_MIN,LEV_MAX,LEV_HIST,
     *  NARC,IAUTOC,NLEV2,NSIGC,CANGLE,IVCOMP,
     *  CONTOUR_FEWER,CENTERED,SET_DELTA,DELTA,NEGINF,POSINF,CONT_KEY,
     *  OPENLEV,SET_STD,SET_MEAN,SET_MIN,SET_MAX,NEED_DATA_SCAN,USE_HIST,
     *  LEV_STR

c
c	zlev()	array of levels to be contoured (user units)
c	ldig()	number opf decimal places in the contour labels
c		=0  no decimal
c		=-2 no label
c	lwgt()	=1 standard line
c		=2 heavy line
c		=3 dashed line
c       lclr()  index of color to use
c	nlev	number of levels to use
c	hgt	height of contour labels (inches)
c	narc	number of subsegments to replace each line segment of 
c		contour.
c	iautoc	=0  use supplied levels
c		=1  compute levels from input data and nlev
c	nlev2	actual number of levels to use 
c	nsigc	number of significant digits for labels if iautoc=1
c	cangle	angle to rotate the contour plot about its center.
c	ivcomp	vector component to contour or surf
c
c 	These added 8.3.95 *jd*
c	centered 	True if levels are to be centered around zero
c	set_delta	True if DZ will be specified by the user
c	delta		The specified DZ value
*       neginf          The low level is open: lowest level specified to -inf
*       posinf          The high level is open: hithest level specified to inf
*       cont_key        Draw continuous color key - no a outline on the key boxes 
*                        (previously this was in shade_vars.cmn)
*	need_data_scan	true if we need to get min/max or mean/std from the data
*       openlev		use open-ended levels
*	set_std		flags: user-set mean and std for open-ended levels
*	set_mean           or user indicated to compute from the data.
*	lev_std		the values
*	lev_mean
*	set_min		flags: user-set min and/or max. if min and OPEN then 
*	set_max            open only on top
*	lev_min		the values of min and/or max.
*	lev_max

c*******************************************************************

