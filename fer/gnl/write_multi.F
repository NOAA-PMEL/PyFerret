	SUBROUTINE WRITE_MULTI( lun, string )

* write multiple lines of the form line//CR//line... to unit lun

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200: 11/27/89
* 	 2/20/90 - corrected string length bug for more-than 2 lines
* V300: adapted for output to GUI displa
* V314: 8/22/94 *kob* IBM port - need to use risc buff for char. concat 
* V320: 2/28/95 - SPLIT_GUI_MESSAGE --> TM_SPLIT_MESSAGE

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#ifdef NO_PASSED_CONCAT
	include 'ferret_cmn/xrisc.cmn'
#endif
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER		lun
	CHARACTER*(*)	string

* internal variable declarations:
	LOGICAL 	multiline
	INTEGER		TM_LENSTR1, cr_pos, c1, clast

* initialize
	c1 = 1
	
* multiple lines ?
 100	cr_pos = INDEX( string(c1:), pCR )
	multiline = cr_pos .NE. 0

* get last character of line
	IF ( multiline ) THEN
	   clast = c1 + cr_pos - 2
	ELSE
	   clast = (c1-1) + TM_LENSTR1( string(c1:) )		! mod 2/90
	ENDIF

* write one line
#ifdef NO_PASSED_CONCAT
	risc_buff = string(c1:clast)
	CALL TM_SPLIT_MESSAGE(lun, '          '//risc_buff)
#else
	CALL TM_SPLIT_MESSAGE(lun, '          '//string(c1:clast))
#endif

* more ?
	IF ( multiline ) THEN
	   c1 = clast + 2
	   GOTO 100
	ENDIF

* no. done
	RETURN
	END
