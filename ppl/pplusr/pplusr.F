	subroutine pplusr(command,ier,msg)
C** 
C**    @(#)pplusr.f	1.1    3/10/88
C**
c
c
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	character command*(*),msg*(*)
	integer   ier
c
	character com*30,args*255
c
	ier=1
	msg='usr command not implemented at this location'
	com='*QG$MSG'
	ilen=lnblk(msg,80)
	call putsym(com,msg,ilen,ier2)
	com='*QG$IER'
	write(args,'(i1)')ier
	call putsym(com,args,1,ier2)
	return
	end
c
