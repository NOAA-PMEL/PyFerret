      SUBROUTINE REDSF(X,NXS,NYS,NSIZE,IBASE,FILE) 
	DIMENSION X(NSIZE)
C** 
C**    @(#)redsf.F	1.3    9/8/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
#ifdef unix
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/LINES.INC'
	INCLUDE 'pplinc/HD.INC'
	INCLUDE 'pplinc/PPL.INC'
	INCLUDE 'pplinc/DATA.INC'
	INCLUDE 'pplinc/CMRD.INC'
	INCLUDE 'pplinc/CMRDL.INC'
	INCLUDE 'pplinc/LABELS.INC'
	INCLUDE 'pplinc/LABCOM.INC'
	INCLUDE 'pplinc/AXISL.INC'
	INCLUDE 'pplinc/TAXIS.INC'
	INCLUDE 'cmdinc/SWITCH.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:LINES.INC'
        INCLUDE 'pplinc:HD.INC'
	INCLUDE 'pplinc:PPL.INC'
        INCLUDE 'pplinc:DATA.INC'
        INCLUDE 'pplinc:CMRD.INC'
        INCLUDE 'pplinc:CMRDL.INC'
        INCLUDE 'pplinc:LABELS.INC'
        INCLUDE 'pplinc:LABCOM.INC'
        INCLUDE 'pplinc:AXISL.INC'
        INCLUDE 'pplinc:TAXIS.INC'
        INCLUDE 'CMDINC:SWITCH.INC'
#endif
	character sym*30,str*120,file*(*)
C     NREC=NUMBER OF RECORDS TO BE READ 
C     NGRP=NUMBER OF GROUPS TO BE READ PER RECORD 
C     NVAR=NUMBER OF VARIABLE PER GROUP 
	IX=IBASE-1
	IY=NSIZE/2+IX
C	SAVE THE REQUESTED NUMBER OF POINTS
	NXT=NX
	LEN=0
	CALL OPNDSF(FILE,'RD',LUNN)
	IF(IFORM(1:3).EQ.'PPL')THEN
c
c	read format pplus dsf data file
c
	    call rdhdsf(lunn,367,ppts,j)
	    if(j.eq.0)goto 12
c
c	set hd values
c
	    nx=pnx
	    ny=pny
	    jtype=ptype
	    xmin=pxmin
	    ymin=pymin
	    xmax=pxmax
	    ymax=pymax
	    zmin=pzmin
	    zmax=pzmax
c
c	transfer plot labels
c
	    write(str,998)pxlab
998	    format(20a4)
	    write(sym,997)'XLAB'
997	    format('*PPL$P_',a)
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
	    if(igtlab.eq.1)xlab=str
c
	    write(str,998)pylab
	    write(sym,997)'YLAB'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
	    if(igtlab.eq.1)ylab=str
c
	    write(str,996)pmlab
996	    format(30a4)
	    write(sym,997)'LAB'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
	    if(igtlab.eq.1)lab1=str
c
c	plot file information
c
	    write(str,995)pdate
995	    format(4a4)
	    write(sym,997)'PDATE'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)ptime
	    write(sym,997)'PTIME'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,994)pident
994	    format(8a4)
	    write(sym,997)'PIDENT'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,994)psrc
	    write(sym,997)'PSOURCE'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
c	variable informationc
c
	    write(str,994)varnme
	    write(sym,997)'VARNAME'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vmean
	    write(sym,997)'VMEAN'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vvar
	    write(sym,997)'VVAR'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vstd
	    write(sym,997)'VSTD'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vrms
	    write(sym,997)'VRMS'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vmin
	    write(sym,997)'VMIN'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vmax
	    write(sym,997)'VMAX'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vdate
	    write(sym,997)'VDATE'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vtime
	    write(sym,997)'VTIME'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vjdate
	    write(sym,997)'VJDATE'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,994)vident
	    write(sym,997)'VIDENT'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vlat
	    write(sym,997)'VLAT'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,995)vlong
	    write(sym,997)'VLONG'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
c	grid information
c
	    write(str,993)gtype
993	    format(2a4)
	    write(sym,997)'GTYPE'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,993)gornt
	    write(sym,997)'GORIENT'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,994)grid
	    write(sym,997)'GRID'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
c	misc comments and user fields
c
	    write(str,996)cmnta
	    write(sym,997)'COMMENT_A'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,996)cmntb
	    write(sym,997)'COMMENT_B'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,996)cmntc
	    write(sym,997)'COMMENT_C'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,996)cmntd
	    write(sym,997)'COMMENT_D'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,996)cmnte
	    write(sym,997)'COMMENT_E'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,994)usera
	    write(sym,997)'USER_A'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,994)userb
	    write(sym,997)'USER_B'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,994)userc
	    write(sym,997)'USER_C'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	    write(str,994)userd
	    write(sym,997)'USER_D'
	    len=lnblk(str,120)
	    call putsym(sym,str,len,ier)
c
	ELSE IF(IFORM(1:3).EQ.'DSF')THEN
	    CALL RDHDSF(LUNN,IDSFRM,XPTS,J)
	    IF(J.EQ.0)GOTO 12
	    WRITE(LAB1,'(16A2)')(JLAB(I),I=1,16)
	    LAB1(NCH+1:)=' '
	    WRITE(XLAB,'(16A2)')(IXLB(I),I=1,16)
	    XLAB(NCHX+1:)=' '
	    WRITE(YLAB,'(16A2)')(IYLB(I),I=1,16)
	    YLAB(NCHY+1:)=' '
	ELSE
	    CALL RDHDSF(LUNN,20,P,J)
	    IF(J.EQ.0)GOTO 12
	    XPTS=P(1)
	    ZMIN=P(2)
	    ZMAX=P(3)
	    ZMEAN=P(4)
	    JTYPE=1
	    NX=IFIX(XPTS)
	    NY=1
	    XMIN=1
	    XMAX=NX
	    YMIN=ZMIN
	    YMAX=ZMAX
C
C	    GET DSF LABELS
C
	    IF(IGTLAB.EQ.1)CALL DSFLAB
	    IF(ITFLG.EQ.1)THEN
		CALL DSFLIM
		CALL MTMDY(P(11),IM,ID,IYR)
		WRITE(ICOM,999)IYR,IM,ID,IFIX(P(13))
999		FORMAT(3I2.2,I4.4)
		OFF=DIFF(ITSTRT,ICOM)/DT+1.
		P(14)=P(14)/DT
	    ELSE
		P(14)=1.0
		OFF=1.0
	    ENDIF
	ENDIF
        if(jtype.eq.2 .or. jtype.eq.-2)then
           if(svectr)then
              jtype=-2
              ivlen=nx*ny+nx+ny
              call rdddsv(lunn,nsize,x,j,ivlen+1)
              iv=ivlen+nx*ny
              call rdddsv(lunn,nsize,x,j,iv+1)
              call rdddsv(lunn,nsize,x,j,iv+nx+1)
           else
              call rdddsv(lunn,nsize,x,j,1)
              iv=nx*ny
              call rdddsv(lunn,nsize,x,j,iv+1)
              call rdddsv(lunn,nsize,x,j,iv+nx+1)
           endif
        else if(jtype.eq.0 .or. jtype.eq.-1)then
	    IF(SVECTR)THEN
                jtype=-1
		CALL RDDDSV(LUNN,NSIZE,X,J,NX*NY+1)
	    ELSE
		CALL RDDDSV(LUNN,NSIZE,X,J,1)
	    ENDIF
	ELSE
	    IF(IFORM(1:4).EQ.'BIBO')THEN
		CALL RDDDSV(LUNN,NXT,X,J,IY+1)
C		SET NX TO ACTUAL NUMBER OF POINTS READ
		NX=J
		DO 2001 J=1,NX
		X(IX+J)=(FLOAT(J)-1.0)*P(14)+OFF
2001		CONTINUE
	    ELSE
C		SET TO READ BIBO FORMAT
		NX=MIN0(NXT,NX)
		CALL RDDDSV(LUNN,NX,X,J,IX+1)
		ISKP=NX-NXT
		IF(ISKP.GT.0)THEN
C		    NEED TO SKIP SOME OF THE X VARIABLE USE Y AS DUMMY
		    CALL RDDDSV(LUNN,ISKP,X,J,IY+1)
		ENDIF
		CALL RDDDSV(LUNN,NX,X,J,IY+1)
	    ENDIF
	ENDIF
	LEN=NX*NY
12	CALL CLSDSF(LUNN)
	LLENG(LINEN)=LEN
	RETURN
	END 
