	SUBROUTINE ALIAS_COMMAND( buffer, clen, * )

* check to see if an alias was used to abbreviate the command and replace with
* the full string if so

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  2/8/90
* V230:  1/27/92 - added QUIT, DATA, and REGION
*           2/92 - improved logic to permit aliases of less than 4 chars
*         6/8/92 - upper bound of aliases changed to reflext DEFINE ALIAS
*        6/19/92 - alias names 8 characters
*        7/29/92 - uses ALIAS_ID
* V300: take abnormal return if alias substitution occurs
* V313: *sh* AIX port.  logic for replacing buffer doesn't work

* calling argument declarations:
	INTEGER	  clen
	CHARACTER buffer*(*)

* internal variable declarations:
	INTEGER	TM_LENSTR1, ALIAS_ID, 
     .          alen, true_len, i, spos, bpos

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcommand.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCOMMAND.CMN'
#endif

#ifdef AIX_XLF
	include 'ferret_cmn/xrisc.cmn'
#endif

* what is the alias command name length - terminated by "/" or " "
        alen = 8
        spos = INDEX(buffer(:alen),'/')
        bpos = INDEX(buffer(:alen),' ') 
        IF ( spos .NE. 0 ) alen = spos - 1
        IF ( bpos .NE. 0 ) alen = MIN( alen, bpos - 1 )
        
* identify the alias
        i = ALIAS_ID( buffer(:alen) ) 

* substitute alias text if appropriate
        IF ( i .NE. unspecified_int4 ) THEN
	   true_len = TM_LENSTR1( alias(i) )
#ifdef AIX_XLF
	   risc_buff = buffer(alen+1:clen)
	   buffer = alias(i)(:true_len)//risc_buff(:clen-alen)
#else
	   buffer = alias(i)(:true_len)//buffer(alen+1:clen)
#endif
	   clen = clen + ( true_len - alen )	   
	   RETURN 1    ! abnormal return if substitution
	ENDIF

* no substitution
	RETURN
	END
