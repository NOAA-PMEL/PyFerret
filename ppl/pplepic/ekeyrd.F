	subroutine ekeyrd (varcd, labsml, label, comment)
C** 
C**    @(#)ekeyrd.F	1.1.1.1    7/30/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Reads EPICKEY file (Unit lun21) to get short graph label, full label and
c  comment for variable code varcd.
c
c  Programmed by N. Soreide, Oct 85.
c
	common /epiclun/ lun1,lun11,lun21
c
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	character varcd*(*), labsml*(*), label*(*), comment*(*)
#ifdef unix
	character varrd*4
#endif
c
	if (lun21 .eq. 0) lun21=21
#ifdef unix
	rewind(lun21)
10	read (lun21, 100,err=200,end=200) varrd, labsml, label, comment
100	format (a4, x, a3, x, a25, x, a97)
	if(varrd.ne.varcd)goto 10
#else
        read (lun21, 100, key=varcd, err=200) labsml, label, comment
100     format (4x, x, a3, x, a25, x, a97)
#endif
	return
c
200	labsml(1:)=' '
	label(1:)=' '
	comment(1:)=' '
	return
	end
