	SUBROUTINE TM_FRIENDLY_READ( prompt, buff, * )

* get a line of input allowing full gnu-readline line editing
* take abnormal return if eof is encountered

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for DECstation computer under Ultrix and VMS operating systems
*
* rev 0.0 -   5/9/91
*  Sun Port 1/30/92 -kob-  added ifdef sun because uses BYTE whereas
*  DECstation uses  INTEGER*1
* 1/94 *sh* increased max read buffer len (blen) from 256 to 512

* calling argument declarations:
        CHARACTER*(*) prompt, buff

#ifdef unix

* internal variable declarations:
        INTEGER plen, blen
        PARAMETER (plen = 8, blen = 512 )
#ifdef sun
        BYTE      hprompt(plen), hbuff(blen)
#else
        INTEGER*1 hprompt(plen), hbuff(blen)
#endif

* translate to null-terminated c-type strings
        CALL TM_FTOC_STRNG( prompt, hprompt, plen )

* call c-written routine to get input line
        CALL TM_FTOC_READLINE( hprompt, hbuff )

* translate recieved input line to FORTRAN character array
        CALL TM_CTOF_STRNG( hbuff, buff, blen )

* null string returned means that EOF (^D) was entered
        IF ( hbuff(1) .EQ. 4 ) RETURN 1     ! ^D passed

#else
        STOP 'TM_FRIENDLY_READ isn't used by VMS'
#endif
	RETURN
	END

