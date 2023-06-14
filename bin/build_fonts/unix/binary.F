	program binary
	character file*120
	integer*2 ascii(128),icnt,ibase,irast,ilen
c
	integer*2 x(5000),y(5000)
c
	integer ij
	call getarg(1,file)
	open(1,file=file,status='old')
	call getarg(2,file)
	if(file.eq.' ')call exit(0)
	open(2,file=file,status='new',form='unformatted')
	read(1,987)icnt,ibase,irast
987	format(1x,i2,2i4)
	if(icnt.eq.99)then
	read(1,985)(ascii(ij),ij=1,128)
	ilen=ascii(128)
	else
	read(1,985)(ascii(ij),ij=1,icnt+1)
	ilen=ascii(icnt+1)
	do ij=icnt+1,128
	  ascii(ij)=1
        enddo
	endif
985	format(1x,20i4)
	read(1,984)(x(ij),ij=1,ilen)
	read(1,984)(y(ij),ij=1,ilen)
984	format(1x,40i2)
	close(1)
	write(2)icnt,ibase,irast,ilen
	write(2)(ascii(ij),ij=1,127)
	write(2)(x(ij),ij=1,ilen)
	write(2)(y(ij),ij=1,ilen)
	close(2)
	end
