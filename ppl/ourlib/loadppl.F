	subroutine ldgrid(xmin,xmax,nx,ymin,ymax,ny,itype,ang,gty,info)
C** 
C**    @(#)loadppl.f	1.3    5/31/88
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
c	load and read grid information into ppl header
c
	character info*(*),gty*(*),str*32
#ifdef unix
	include 'pplinc/PPL.INC'
#else
	include 'pplinc:PPL.INC'
#endif
c
	pxmin=xmin
	pxmax=xmax
	pnx=nx
	pymin=ymin
	pymax=ymax
	pny=ny
	ptype=itype
	gxornt=ang
	write(str,'(f5.1)')ang
	read(str,999)gornt
999	format(2a4)
	read(gty,999)gtype
	read(info,998)grid
998	format(8a4)
	return
	entry gtgrid(xmin,xmax,nx,ymin,ymax,ny,itype,ang,gty,info)
c
	xmin=pxmin
	xmax=pxmax
	nx=pnx
	ymin=pymin
	ymax=pymax
	ny=pny
	itype=ptype
	ang=gxornt
	write(gty,999)gtype
	write(info,998)grid
	return
	end
c
	subroutine ldcmnt(cma,cmb,cmc,cmd,cme)
c
c	load the 120 character comments for the ppl header
c
	character cma*(*),cmb*(*),cmc*(*),cmd*(*),cme*(*)
#ifdef unix
	include 'pplinc/PPL.INC'
#else
	include 'pplinc:PPL.INC'
#endif
c
	read(cma,999)cmnta
999	format(30a4)
	read(cmb,999)cmntb
	read(cmc,999)cmntc
	read(cmd,999)cmntd
	read(cme,999)cmnte
	return
	entry gtcmnt(cma,cmb,cmc,cmd,cme)
	write(cma,999)cmnta
	write(cmb,999)cmntb
	write(cmc,999)cmntc
	write(cmd,999)cmntd
	write(cme,999)cmnte
	return
	end
c
	subroutine lduser(usa,usb,usc,usd)
c
c	load the 32 character user strings into the ppl header
c
	character usa*(*),usb*(*),usc*(*),usd*(*)
#ifdef unix
	include 'pplinc/PPL.INC'
#else
	include 'pplinc:PPL.INC'
#endif
c
	read(usa,999)usera
999	format(8a4)
	read(usb,999)userb
	read(usc,999)userc
	read(usd,999)userd
	return
	entry gtuser(usa,usb,usc,usd)
	write(usa,999)usera
	write(usb,999)userb
	write(usc,999)userc
	write(usd,999)userd
	return
	end
c
	subroutine ldplid(ident,source,dte,tme)
c
c	load the plot identification information for the ppl header
c
	character ident*(*),source*(*),dte*(*),tme*(*)
#ifdef unix
	include 'pplinc/PPL.INC'
#else
	include 'pplinc:PPL.INC'
#endif
c
	read(ident,999)pident
999	format(8a4)
	read(source,999)psrc
	read(dte,998)pdate
998	format(4a4)
	read(tme,998)ptime
	return
	entry gtplid(ident,source,dte,tme)
	write(ident,999)pident
	write(source,999)psrc
	write(dte,998)pdate
	write(tme,998)ptime
	return
	end
c
	subroutine ldvar(name,dte,tme,ident,
     *		mean,var,std,rms,min,max,jdate,lat,long)
c
c	load the variable information into the ppl header
c
	character name*(*),dte*(*),tme*(*),ident*(*),str*16
	real mean,var,std,rms,min,max,jdate,lat,long
#ifdef unix
	include 'pplinc/PPL.INC'
#else
	include 'pplinc:PPL.INC'
#endif
c
	vxmean=mean
	vxvar=var
	vxmin=min
	vxmax=max
	vxjdat=jdate
	vxlat=lat
	vxlong=long
	read(name,999)varnme
999	format(8a4)
	read(dte,998)vdate
998	format(4a4)
	read(tme,998)vtime
	read(ident,999)vident
c
c	format the variables
c
	write(str,1000)mean
1000	format(1pg12.5)
	read(str,998)vmean
	write(str,1000)var
	read(str,998)vvar
	write(str,1000)std
	read(str,998)vstd
	write(str,1000)rms
	read(str,998)vrms
	write(str,1000)min
	read(str,998)vmin
	write(str,1000)max
	read(str,998)vmax
	write(str,1001)jdate
1001	format(f8.2)
	read(str,998)vjdate
	ideg=abs(int(lat))
	imin=(abs(lat)-ideg)*60
	write(str,1002)ideg,imin
1002	format(i3,'#',i3,' ')
	k=8
	if(imin.eq.0)k=4
	if(lat.gt.0)then
	    str=str(:k)//'N'
	else
	    str=str(:k)//'S'
	endif
	read(str,998)vlat
	ideg=abs(int(long))
	imin=(abs(long)-ideg)*60
	write(str,1002)ideg,imin
	k=8
	if(imin.eq.0)k=4
	if(long.gt.0)then
	    str=str(:k)//'E'
	else
	    str=str(:k)//'W'
	endif
	read(str,998)vlong
	return
	entry gtvar(name,dte,tme,ident,
     *		mean,var,std,rms,min,max,jdate,lat,long)
c
c	not all the formatted strings are returned here!
c
	mean=vxmean
	var=vxvar
	min=vxmin
	max=vxmax
	jdate=vxjdat
	lat=vxlat
	long=vxlong
	write(name,999)varnme
	write(dte,998)vdate
	write(tme,998)vtime
	write(ident,999)vident
	return
	end
c
	subroutine ldaxlb(xlab,ylab,mlab)
c
c	load the axis labels into the ppl header
c
	character xlab*(*),ylab*(*),mlab*(*)
#ifdef unix
	include 'pplinc/PPL.INC'
#else
	include 'pplinc:PPL.INC'
#endif
c
	read(xlab,999)pxlab
999	format(20a4)
	read(ylab,999)pylab
	read(mlab,998)pmlab
998	format(30a4)
	return
	entry gtaxlb(xlab,ylab,mlab)
	write(xlab,999)pxlab
	write(ylab,999)pylab
	write(mlab,998)pmlab
	return
	end
