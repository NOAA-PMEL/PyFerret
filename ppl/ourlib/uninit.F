	subroutine uninit(pname,switch,sflag,smap,ns,
     *	    name,nmap,nin,nout)
C** 
C**    @(#)uninit.f	1.1    3/10/88
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
c	subroutine uses iargc and getarg to determine what switches 
c	have been set
c
c	pname	- calling program name (as called) char
c	switch	- list of legal switches char*1
c	sflag	- value of switch .true. if set
c	smap	- points to name associated with switch int
c	ns	- number of switches
c	name	- list for non-switch arguements char(nin)*(*)
c	nmap	- points to the switch associated with name
c	nin	- length of name
c	nout	- number of name at output
c
c
	character pname*(*),switch(ns)*1,name(nin)*(*),arg*255
	logical sflag(ns)
	integer smap(ns),nmap(nin)
c
c
	do 10 i=1,ns
	    sflag(i)=.false.
	    smap(i)=0
10	continue
	do 20 i=1,nin
	    name(i)=' '
	    nmap(i)=0
20	continue
c
c	get program name and number of args
c
	call getarg(0,pname)
	iarg=iargc()
c
c	parse arguement list
c
	nout=0
	ilast=0
	do 30 i=1,iarg
	    call getarg(i,arg)
	    if(arg(1:1).eq.'-')then
		k=lnblnk(arg)
		do 40 kk=2,k
		    do 50 j=1,ns
			if(arg(kk:kk).eq.switch(j))then
			    sflag(j)=.true.
			    ilast=j
			    goto 40
			endif
50		     continue
40		continue
	    else
		nout=nout+1
		if(nout.gt.nin)goto 30
		if(ilast.ne.0)then
		    smap(ilast)=nout
		    nmap(nout)=ilast
		    ilast=0
		endif
		name(nout)=arg
	    endif
30	continue
	return
	end
