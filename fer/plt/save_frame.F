	SUBROUTINE SAVE_FRAME( filename, format, status )

* make a movie or GIF frame of the current output graphics

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  11/14/89 - based on XEQ_FRAME
* V320: 5/94 - array "memory" as a calling argument
*	8/94 - filename added as calling argument
*	1/95 - major revamp using entirely X code instead of GKS (much faster)
*		and incorporating GIF output
*       9/97 - now supports batch mode

* calling argument declarations:
	INTEGER		status
	CHARACTER*(*)	filename, format

* internal variable declarations:
	INTEGER		TM_LENSTR1, i, elen, flen, wsid
	CHARACTER	errstr*128
	logical         ITS_BATCH_GRAPHICS

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/movies.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:MOVIES.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
#endif

        INTEGER slen
        PARAMETER  ( slen = 128 )

* declarations for C-calling machinery
#ifdef sun
        BYTE      fnmhol(slen), errhol(slen), fmthol(5) ! c Hollerith strings
#else
        INTEGER*1 fnmhol(slen), errhol(slen), fmthol(5) ! cHollerith strings
#endif

* translate to null-terminated c-type strings
	flen = TM_LENSTR1(filename)
	CALL TM_FTOC_STRNG( filename(:flen), fnmhol, slen )
	CALL TM_FTOC_STRNG( format, fmthol, 5 )

* determine the current output window (GKS wkstn ID, that is)
	DO wsid = 1, max_windows
	   IF ( wn_active(wsid) ) GOTO 110
	end do

* blank out the error message return buffer (a C string)
 110	DO i = 1, slen
	   errhol(i) = 0	   
	end do

* if running in batch mode, take care of metafiles rather than
* dumping window
	if (ITS_BATCH_GRAPHICS()) then
	   call put_frame_batch(wsid, fnmhol, fmthol, errhol, status)
	else
* capture the bitmap (C routine)
	   CALL PUT_FRAME( wsid, fnmhol, errhol, fmthol, status )
	end if

* illegal filename errors in status, bitmap capture errors in errstr
	IF ( status .LT. 0 ) THEN
	   GOTO 5200
	ELSEIF ( errhol(1) .NE. 0 ) THEN
	   CALL TM_CTOF_STRNG( errhol, errstr, slen )
	   elen = TM_LENSTR1(errstr)
	   GOTO 5300
	ENDIF

* successful completion
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5200	CALL ERRMSG( ferr_erreq, status, 'Unable to open GIF file: '
     .		     //filename(:flen), *5000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .               errstr(:elen), *5000 )

	END

