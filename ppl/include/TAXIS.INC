c
c	taxis
C** 
C**    @(#)TAXIS.INC	1.2    3/15/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
* V694  *acm*  1/15 Changes to allow two time axes on plot (F-T)
* V695  *acm*  9/15 Changes for hour:min labeling on time axes
* V697  *acm*  1/16 Ticket 2344: if there was a time-plot underlay, use 
*                   its delta-t for time overlay plots, and clear in disp_init

	COMMON/TXSCOM/ TLO,DT,DT_SAVE,TCSIZE,ITYPET, 
     *	NMTCT,LABELT,IFDB,ISDB,ITFLG,VANG,IVFLG,IAUTOT,
     *	ITXTPE,ITXLBP,TLOb,DTb,TCSIZEb,ITYPETb, 
     *	NMTCTb,LABELTb,IFDBb,ISDBb,ITFLGb,VANGb,IVFLGb,IAUTOTb,
     *	ITXTPEb,ITXLBPb,is_hours_axis
c
c	tlo	x value corresponding to the start time (index number)
c	dt	sampling rate in minutes (minutes)
c	dt_save	sampling rate in minutes (minutes) from the current underlay plot
c	tcsize	character height for time axis labels (inches)
c	itypet	=0 use taxis0  hours/days
c		=1 use taxis1  days/months
c		=3 use taxis3  months/years
c	nmtct	number of small tics
c	labelt	position of labels and label type control
c	ifdb	labeling interval primary labels
c	isdb	labeling interval secondary labels
c	itflg	=-1 y time axis on
c		=0    time axis off
c		=1  x time axis on
c		=2  x and y time axis on, F-T plot.
c	vang	rotation angle for stick plots (degrees)
c	ivflg	=0 normal line plot
c		=1 PLOTV command used
c		=2 PLOTUV command used
c		=3 VELVCT command used
c	iautot	=0 manual entry of time axis specifications
c		=1 automatic selection of time axis specs (EPIC and BIBO only)
c
c	itxtpe  the type of label
c
c	itxlbp  the position of the label  labelt=itxtpe*itxlbp
c
c*******************************************************************
