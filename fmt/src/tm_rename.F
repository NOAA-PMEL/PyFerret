	SUBROUTINE TM_RENAME ( oldname, newname, status )

* rename a file

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for DECstation computer under Ultrix and VMS operating systems
*
* rev 0.0 -   5/6/91

* calling argument declarations:
        INTEGER  status
        CHARACTER*(*) oldname, newname

* internal variable declarations:
	INTEGER	olen, nlen
        INTEGER TM_LENSTR1

#ifdef unix
        include 'tmap_format/tmap_errors.parm'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

#ifdef unix
        INTEGER slen
        PARAMETER  ( slen = 128 )
        INTEGER oldhol(slen), newhol(slen)  ! c-type Hollerith string buffers

* initialize
        olen = TM_LENSTR1(oldname)
        nlen = TM_LENSTR1(newname)

* translate to null-terminated c-type strings
        CALL TM_FTOC_STRNG( oldname(:olen), oldhol, slen )
        CALL TM_FTOC_STRNG( newname(:nlen), newhol, slen )

* call c-written routine to make system call
        CALL TM_C_RENAME( oldhol, newhol, status )

* translate Unix system error code to TMAP library form
        IF ( status .EQ. 0 ) THEN
           status = merr_ok
        ELSE
           status = merr_erreq
        ENDIF

#else
        STOP 'TM_RENAME for VMS isn't yet written'
!    ... insert VMS system call here

#endif
	RETURN
	END

