	subroutine getlev(z,nxs,nys,nx,ny,dz,zlev,nlev) 
C** 
C**    @(#)getlev.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c     given the array z(1...nx,1...ny) and the contour interval dz 
c     the array zlev(1...nlev) of contour levels is returned. 
c 
	dimension z(nxs,nys) 
	dimension zlev(2) 
	big=.9e35 
c 
	nfnd=0 
	do 100 i=1,nx 
	do 100 j=1,ny 
	if(z(i,j)-big)70,100,100 
70	nfnd=nfnd+1 
	if(nfnd-1)80,80,90 
80	zmin=z(i,j) 
	zmax=zmin 
90	zmin=amin1(zmin,z(i,j)) 
	zmax=amax1(zmax,z(i,j)) 
100	continue 
c 
	nlmin=zmin/dz+.5+sign(.5,zmin) 
	zlev(1)=dz*nlmin 
	do 200 k=1,51 
	zlevk=zlev(1)+(k-1)*dz 
	if(zlevk-zmax)190,210,210 
190	zlev(k)=zlevk 
200	continue 
210	nlev=k-1 
	return 
	end 
