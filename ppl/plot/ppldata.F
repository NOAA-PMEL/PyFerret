	BLOCK DATA PPLDAT
*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
C**    @(#)ppldata.F	1.2    5/31/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C	LIST OF INCLUDES FOR PPLUS
C
C     Mod *JD* 3.5.91 Change output unit for terminal to 6 from 5 --
C     problem with buffered outout in unix -- see opnppl
C     Mod *jd* 1.10.95 To DATA KEYLN1,KEYLN2 (line 201) at *sh* request
C     for compatibility with Ferret use of symbols.
* V510 4/99 *jd* increased initialization sizes for bigger NLINES
* V510 9/99 *sh* increased initialization sizes for bigger NLINES
*      3/00 *sh* reduceded initialization sizes for NLINES variables
* 		(except lleng())
* V610 2/08 *acm*increase default number of contour levels to NLEV=40 (MODE NLEVELS)
* V610 2/08 *acm*keep default number of contour levels to old value of NLEV=10 
*                If this is changed, also change it in fer/dat/xprog_state_data.F
* V62  2/08 *acm*increase default number of contour levels to NLEV=30 (MODE NLEVELS)
*                If this is changed, also change it in fer/dat/xprog_state_data.F
* v691 5/14 *acm* Change name MOD to TEKMOD
* V694  *acm*  1/15 Changes to allow two time axes on plot (F-T)
* v696 *acm*	gap_var to define seperate line segments on PLOT/VS plots
*                  (LAS ticket 1597 and others)
* V697 *acm*   Fix CONSET for setting # of digits on contour plots. When did this ever work?
* V73+ 1/18 *acm* For Issue 1009; precsision in time info sent to pplus for 2D plots, new iautot_save

	INCLUDE 'PARAMPL5.DAT'
	include 'axis_inc.decl'
	INCLUDE 'AXIS.INC'
	INCLUDE 'AXISL.INC'
	INCLUDE 'AXLWID.INC'
	INCLUDE 'CMRD.INC'
	INCLUDE 'CMDCOM.INC'
	INCLUDE 'CMRDL.INC'
	INCLUDE 'COMEPL.INC'
	INCLUDE 'COMEPV.INC'
	INCLUDE 'CONLAB.INC'
	INCLUDE 'cont_inc.decl'
	INCLUDE 'CONT.INC'
	INCLUDE 'DASHZZ.INC'
	INCLUDE 'DATA.INC'
	INCLUDE 'EPICLUN.INC'
	INCLUDE 'FILNAM.INC'
	INCLUDE 'GKSCM1.INC'
	INCLUDE 'HD.INC'
	INCLUDE 'LABCOM.INC'
	INCLUDE 'LABELS.INC'
	INCLUDE 'LINES.INC'
	INCLUDE 'LUNITS.INC'
	INCLUDE 'MISS.INC'
	INCLUDE 'PEN.INC'
	INCLUDE 'PLT.INC'
	INCLUDE 'PLTL.INC'
	INCLUDE 'PPLDAT.INC'
	INCLUDE 'SWITCH.INC'
	INCLUDE 'SYMKEY.INC'
	INCLUDE 'SYSTEM.INC'
	INCLUDE 'TAXIS.INC'
	INCLUDE 'TICS.INC'
	INCLUDE 'TICS2.INC'
	INCLUDE 'XYLABP.INC'
	INCLUDE 'vector_inc.decl'
	INCLUDE 'VECTOR.INC'
	INCLUDE 'VIEW.INC'
	INCLUDE 'ZGRID.INC'
	INCLUDE 'ZGRIDD.INC'
C
	DATA XLO,XHI,XTIC,XLEN,XCSIZE,IAUTOX,ITYPEX/3*0.,8.,.10,1,1/
	DATA NMTCX,LINTX,LABELX,NSIGX,NTICX,XORG/0,2,-1,2,5,1.4/
	DATA YLO,YHI,YTIC,YLEN,YCSIZE,IAUTOY,ITYPEY/3*0.,6.0,.10,1,1/
	DATA NMTCY,LINTY,LABELY,NSIGY,NTICY,YORG/0,2,-1,2,5,1.2/
	DATA IAXON/4*1/
	DATA SHRINKY/.FALSE./
C
	DATA IFRX,IFRY,IT1,IT2,ITSTRT,ITMIN,ITMAX/2*' ',
     *	5*'71010100000019'/ 
C
	DATA P,M,N,IFLG/20*0.,22*0/ 
C
	DATA ECHLUN,CMDLUN/23,24/
C
	DATA ICOM,LABEL,IFORM,STRING/2*' ','(3F10.2)',' '/
C
	DATA IEFLG/0/
C
	DATA CONPRE,CONPST/' ',' '/
C
C
c*	DATA ZLEV,LDIG,LWGT,LCLR,NLEV,HGT/500*0.0,1500*0,40,.08/
c*	DATA ZLEV,LDIG,LWGT,LCLR,NLEV,HGT/500*0.0,1500*0,10,.08/
	DATA ZLEV,LDIG,LWGT,LCLR,NLEV,HGT/500*0.0,1500*0,30,.08/
	DATA NARC,IAUTOC,NLEV2,NSIGC,INISIG,NEWSIG,CANGLE,IVCOMP/1,1,0,2,2,0,0.0,1/
C
	DATA DASHLN,SPACLN,DSLAB/.04,.04,5.0/
C
	DATA ITYPEZ,JX,JY,JZ,LUNN,LINEN,NREC/0,1,2,0,1,0,1/
	DATA NVAR,NGRP,CAY,NRNG/2,1,5.0,5/
C
	DATA LUN1,LUN11,LUN21/1,11,21/
C
	DATA WSID,CONID,WSTYPE,ERRFIL/1,0,0,0/
	DATA CHDEV,CHPET,LCDEV,LCPET/1,1,1,3/
	DATA WKSAUT/.TRUE./
C
	DATA XPTS,ZMIN,ZMAX,ZMEAN,XMIN,XMAX/6*0.0/
	DATA NX,YMIN,YMAX,NY,JTYPE/1,2*0.0,1,1/
C
	DATA HLAB1,HXLAB,HYLAB,HLABS/.16,3*.12/ 
	DATA ICOUNT,NLAB,ICROSS,IGTLAB/3*0,1/
C
C	FOR NLABS = 50
C
	DATA LAB1,XLAB,YLAB,LABS/3*' ',50*' '/
C
C	FOR NLINES=200 ! 3/00 *sh*
C
	DATA IMARK/0,2,4,18,20,22,26,28,30,48,190*0/
	DATA HMARK,ITYPE/200*0.08,0,9*4,190*0/
	DATA XOFF,YOFF,XFCT,YFCT/201*0.,201*0.,201*1.0,
     .						   201*1.0/
	DATA DN1/3*.15,.05,3*.15,.2,.1,.1,190*.15/
	DATA UP1/3*.1,.05,.05,.15,.05,.1,.05,.1,190*.1/
	DATA DN2/.15,.15,5*.05,3*.1,190*.15/
	DATA UP2/3*.1,.05,.15,.05,.05,.1,.05,.1,190*.1/
	DATA LLENG,LNUM,IWIND/100000*0,1,0/   ! lleng(NPOLYGON)
	DATA IPEN,IP/201*1,0,1,2*0/
	DATA gap_var/0/

C
C       Mod *JD* 3.91 to set default tty output to unit 6, not 5
C	DATA LTTOUT,LMULT/5,6/ original
	DATA LTTOUT,LMULT/6,5/
C
	DATA CMXLE,CMXEQ,CMXGE/0.,0.,1.E35/
	DATA XLE,XEQ,XGE/.FALSE.,.FALSE.,.TRUE./
	DATA CMYLE,CMYEQ,CMYGE/0.,0.,1.E35/
	DATA YLE,YEQ,YGE/.FALSE.,.FALSE.,.TRUE./
	DATA CMZLE,CMZEQ,CMZGE/0.,0.,1.E35/
	DATA ZLE,ZEQ,ZGE/.FALSE.,.FALSE.,.TRUE./
C
	DATA ICODE,WIDTH,HEIGHT,IR,LUN/
     .		3,10.0,8.0,110,8/
C
	DATA IBASE,PBUF/1,0/
C
	DATA NFLAGS/20/
	DATA SWITCH/'ECHO', 'DEBUG', 'OVERLAY','ERASE',
     *		    'WAIT','CENTER',   'RIGHT', 'LEFT',
     *		    'USER', 'LATCH',   'QUIET',  'LOG',
     *		   'YAXIS',  'TIME',  'OFFSET',  'CTD',
     *		    'NEXT','VECTOR',    'LINE',  'FILL'/
	DATA SMIN/       2,       1,         2,      2,
     *			 1,       1,         1,      2,
     *			 1,       2,         1,      2,
     *		         1,	  1,         2,      1,
     *			 2,       2,       2*0/
	DATA DEFLTS/.FALSE.,  .FALSE.,  .FALSE., .TRUE.,
     *		     .TRUE.,  .FALSE.,  .FALSE., .TRUE.,
     *		     .TRUE.,  .FALSE.,  .FALSE.,.FALSE.,
     *		    .FALSE.,  .FALSE.,   .TRUE.,.FALSE.,
     *		    .FALSE.,  .FALSE.,2*.FALSE./
C
C*** Mod 1.10.95 *** DATA KEYLN1,KEYLN2/21,22/
	DATA KEYLN1,KEYLN2/45,46/
C
	DATA ECHOF,DEBUGF,SMKEY2,LOGITF/.TRUE.,.FALSE.,.FALSE.,.FALSE./
	DATA QUIETF,LOGCMF/.FALSE.,.FALSE./
C
	DATA TLO,DT,TCSIZE,ITYPET/1.,1440.,.10,3/
	DATA NMTCT,LABELT,IFDB,ISDB,ITFLG/0,-1,1,1,0/
	DATA VANG,IVFLG,IAUTOT,IAUTOT_save/0.,0,1,1/ 
	DATA TLOb,DTb,TCSIZEb,ITYPETb/1.,1440.,.10,3/
	DATA NMTCTb,LABELTb,IFDBb,ISDBb,ITFLGb/0,-1,1,1,0/
	DATA VANGb,IVFLGb,IAUTOTb/0.,0,1/ 
C
	DATA TSMX,TLX,TSMY,TLY,IX,IY/.125,.25,.125,.25,0,0/ 
C
	DATA ITX,ITY/2*-1/
C
C	FOR NLABS = 50
C
	DATA XLABS,YLABS,RLABS/50*1.E35,100*0.0/
	DATA LTYPE,XLLABS,YLLABS,LLTYPE/50*-1,100*0.0,50*0/
	DATA RHLABS/50*-1.0/
C
	DATA VAUTO,VUSER,VKEY/.TRUE.,.TRUE.,.FALSE./
	DATA VLEN/0.5/
	DATA VFRMT/'(1PG10.3)'/
	DATA VPOS/-1/,VXLOC,VYLOC/0.,0./
	DATA VSKPX,VSKPY/1,1/
	DATA VMINLN,ARMAX,ARMIN,HFACT/.01,0.5,0.1,0.25/
C
	DATA IZTYP,IZMRK,IZLEN/3*0/
C
	END
