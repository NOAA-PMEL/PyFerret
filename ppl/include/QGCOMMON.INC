c
c	variable definitions
C** 
C**    @(#)QGCOMMON.INC	1.1    3/10/88
C**
c
	integer QGPSI,QGZETA,QGRELIEF,QGTOP,QGBOT,QGVEL,QGSPD
	integer QGTEMP,QGPPSI,QGPZETA
c
	parameter (QGPSI=1,QGZETA=2,QGPPSI=3,QGPZETA=4,QGRELIEF=6,
     *		QGTOP=7,QGBOT=8,QGVEL=9,QGSPD=10,QGTEMP=11,
     *		QGMXDLD=20,QGMXDLB=21,QGMXDLT=22,QGMXGLD=25,
     *		QGSBLW=32,QGTRCR1=51,QGTRCR2=52,QGTRCR3=53)
c
c	plot file header storage
c
	parameter (k0=6,k01=7,k02=8)
c
	common/qghead/runid(6),restid(9),
     *	titl(20,2),
     *	tstart,tmax,dt,
     *	m,n,kz,xbasin,theta,
     *	alpha,beta,rf,
     *	iltord,iltfreq,iltcnt,
     *	ifdiff,ifpert,ifrst,iftop,ifbot,iftvv,ifrel,
     *	rlat0,rlng0,t0,v0,dhor,ht,time0,r0,
     *	hz(k0),
     *	rhosur,scsig,sigz(k01),eigval(k0),ainv(k0,k0),amat(k0,k0),
     *	tsurf,xc(k01),
     *	salsur,eta(k01)
c
c
	common/qgcom/openf,hlev(0:k01),tscale,xscale,yscale,xmin,ymin,
     *	next,dtime,time,tstep,level,depth,field,ttyp,dtyp,iqgln,
     *	rtime,leof,timeold,first,tbar(0:k01),sbar(0:k01),dlev(0:k0),
     *	actdep,exclud(53),llgrd,idxgrd
c
	integer*4 rtime
	integer ttyp,dtyp,field
	logical next,openf,leof,first,exclud,llgrd,idxgrd
c
	common/qgstr/qfile,qhdr1,qhdr2,qhdr3,qtitle1,qtitle2,
     *	qvars,qlabel,qlevel,qdate
c
	character qfile*255,qhdr1*120,qhdr2*120,qhdr3*120
	character qtitle1*80,qtitle2*80,qvars*255
	character qlabel*80,qlevel*30,qdate*30
c
	common/qgpntr/psi(0:k01),zeta(0:k01),ppsi(0:k01),pzeta(0:k01),
     *	sblw(50),trc1(50),trc2(50),trc3(50),sblwd(50),trc1d(50),
     *	trc2d(50),trc3d(50),wcount,t1count,t2count,t3count,top,bottom,
     *	mxdld,mxdlb,mxdlt,mxgld
c
	integer psi,zeta,ppsi,pzeta,sblw,trc1,trc2,trc3,wcount,t1count
	integer t2count,t3count,top,bottom,mxdld,mxdlb,mxdlt,mxgld
	real sblwd,trc1d,trc2d,trc3d
c
