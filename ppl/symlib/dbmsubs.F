	subroutine dbmopen(file,flen,lun)
C** 
C**    @(#)dbmsubs.f	1.2    5/26/88
C**
c**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	character file*(*)
	integer flen,lun
c
	return
	end
c
	subroutine dbmstore(lun,sym,ilen,value,len,ier)
c
	integer VSHORT,VLONG
	parameter (VSHORT=1000,VLONG=100)
c
	character sym*(*),value*(*)
	integer lun,ilen,len,ier
c
c	have two regions of storage long (255 characters)
c	and short (132 characters),  this should minimize
c	space requirements on a fortran system
c
	character syml(VLONG)*30,syms(VSHORT)*30
	character vall(VLONG)*255,vals(VSHORT)*132
	integer lunl(VLONG),luns(VSHORT)
	integer lenl(VLONG),lens(VSHORT)
	logical long
c
	data lunl/VLONG*-1/,luns/VSHORT*-1/
	data mxlong/1/,mxshort/1/
c
c	look for sym in LONG list
c
	ier=0
	do 10 i=1,mxlong
	    if(lunl(i).eq.lun)then
		if(syml(i).eq.sym)then
		    vall(i)=value
		    lenl(i)=len
		    return
		endif
	    endif
10	continue
c
c	look for sym in SHORT list
c
	do 20 i=1,mxshort
	    if(luns(i).eq.lun)then
		if(syms(i).eq.sym)then
		    if(len.le.132)then
			vals(i)=value
			lens(i)=len
			return
		    else
c
c	insert into long list
c
			luns(i)=-1
		    endif
		endif
	    endif
20	continue
c
c	sym not found in any list
c
	if(len.le.132)then
	    do 30 i=1,mxshort+1
		if(luns(i).eq.-1)then
		    luns(i)=lun
		    syms(i)=sym
		    vals(i)=value
		    lens(i)=len
		    mxshort=max0(i,mxshort)
		    if(mxshort.gt.VSHORT)stop 'exceeded symbol storage'
		    return
		endif
30	    continue
	else
	    do 40 i=1,mxlong+1
		if(lunl(i).eq.-1)then
		    lunl(i)=lun
		    syml(i)=sym
		    vall(i)=value
		    lenl(i)=len
		    mxlong=max0(i,mxlong)
		    if(mxlong.gt.VLONG)stop 'exceeded symbol storage'
		    return
		endif
40	    continue
	endif
	return
c
	entry dbmdelete(lun,sym,ilen,ier)
c
	ier=0
	do 50 i=1,mxlong
	    if(lunl(i).eq.lun)then
		if(syml(i).eq.sym)then
		    lunl(i)=-1
		    return
		endif
	    endif
50	continue
c
	do 60 i=1,mxshort
	    if(luns(i).eq.lun)then
		if(syms(i).eq.sym)then
		    luns(i)=-1
		    return
		endif
	    endif
60	continue
	return
c
	entry dbmfetch(lun,sym,ilen,value,len,ier)
c
	ier=0
	do 70 i=1,mxlong
	    if(lunl(i).eq.lun)then
		if(syml(i).eq.sym)then
		    value=vall(i)
		    len=lenl(i)
		    return
		endif
	    endif
70	continue
c
	do 80 i=1,mxshort
	    if(luns(i).eq.lun)then
		if(syms(i).eq.sym)then
		    value=vals(i)
		    len=lens(i)
		    return
		endif
	    endif
80	continue
	len=0
	ier=1
	return
c
	entry dbmfirstkey(lun,sym,ilen,ier)
c
	ier=0
	do 90 i=1,mxlong
	    if(lunl(i).eq.lun)then
		sym=syml(i)
		ilen=lnblk(sym,30)
		inext=i+1
		long=.true.
		return
	    endif
90	continue
c
	do 100 i=1,mxshort
	    if(luns(i).eq.lun)then
		sym=syms(i)
		ilen=lnblk(sym,30)
		inext=i+1
		long=.false.
		return
	    endif
100	continue
	ier=1
	ilen=0
	return
c
	entry dbmnextkey(lun,sym,ilen,ier)
c
	ier=0
	if(long)then
	    do 110 i=inext,mxlong
		if(lunl(i).eq.lun)then
		    sym=syml(i)
		    ilen=lnblk(sym,30)
		    inext=i+1
		    long=.true.
		    return
		endif
110	    continue
	    long=.false.
	    do 115 i=1,mxshort
		if(luns(i).eq.lun)then
		    sym=syms(i)
		    ilen=lnblk(sym,30)
		    inext=i+1
		    return
		endif
115	    continue
	    ilen=0
	    ier=1
	else
	    do 120 i=inext,mxshort
		if(luns(i).eq.lun)then
		    sym=syms(i)
		    ilen=lnblk(sym,30)
		    inext=i+1
		    return
		endif
120	    continue
	endif
	ilen=0
	ier=1
	return
c
	entry dbmclose(lun)
	return
c
	entry dbmclear(lun)
c
	do 130 i=1,mxlong
	    if(lunl(i).eq.lun)lunl(i)=-1
130	continue
c
	do 140 i=1,mxshort
	    if(luns(i).eq.lun)luns(i)=-1
140	continue
	return
	end
