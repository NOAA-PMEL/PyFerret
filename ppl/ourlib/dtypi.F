	SUBROUTINE DTYPI(IN,OUT)
C** 
C**    @(#)dtypi.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	INTEGER OUT
	character LIST(48)*2,inc*2
	DATA LIST/'WJ','WI','WA','WC','AJ','AI','TJ','TI',
     *	'AP','WB','DB','AS','AD','  ','AU','AV','TX','TY',
     *	'SL','AT','RS','SP','RD','WT','WU','WV','WP','TL','  ','  ',
     *	'SA','CO','ID','ST','XX','YY','MA','AA','AC','TA','TC','TD',
     *	'TR','AS','ZF','PF','HC','DH'/
	K=IN-11
	read(list(k),999)out
999	format(a2)
	RETURN
	ENTRY IDTYP(IN,OUT)
	write(inc,999)in
	DO 10 I=1,48
	IF(INc.EQ.LIST(I))GOTO 11
10	CONTINUE
	RETURN
11	OUT=I+11
	RETURN
	END
