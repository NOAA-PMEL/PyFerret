	subroutine ekeyopn
C** 
C**    @(#)ekeyopn.F	1.1.1.1    7/30/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Opens EPICKEY file on Unit lun21. Contains CTD Cast Data Header variable
c  codes.
c
	common /epiclun/ lun1,lun11,lun21
c
	if (lun21 .eq. 0) lun21 = 21
c
#ifdef unix
	OPEN (UNIT=LUN21, 
     *	FILE='/usr/local/lib/epic/epickey.values',
     *	FORM='FORMATTED', STATUS='OLD')
#else
        OPEN (UNIT=LUN21, FILE='DISK1:[NNS.EPIC]EPICKEY.DAT',
     1 ACCESS='KEYED', FORM='FORMATTED', KEY=(1:4:CHARACTER),
     2 ORGANIZATION='INDEXED', RECL=132, RECORDTYPE='VARIABLE',
     3 SHARED, READONLY, STATUS='OLD', CARRIAGECONTROL='LIST')
#endif
C
	return
	end
