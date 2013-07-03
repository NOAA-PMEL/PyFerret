*
* Set the name of the graphics engine for windows created
* after this call. 
*
* Input Arguments:
*     windowid - window number to use
*     newname - name (or partial name) of the engine;
*               standard recognized names are Cairo, 
*               CAIRO, PipedViewerPQ, PIPEDVIEWERPQ, 
*               PipedImager, and PIPEDIMAGER
*               Need all-uppercase for /ENGINE="..."
*     forceit - change the engine name regardless of 
*               no-display command-line options?
* Output Arguments:
*     sts - FERR_OK if successful; 
*           FERR_INVALID_COMMAND otherwise
*
        SUBROUTINE FGD_SET_ENGINE(windowid, newname, forceit, sts)

        IMPLICIT NONE
        CHARACTER*(*) newname
        INTEGER windowid, sts
        LOGICAL forceit

        INCLUDE 'ferret.parm'
        INCLUDE 'errmsg.parm'
        INCLUDE 'fgrdel.cmn'

*       Function declarations
        INTEGER TM_LENSTR

*       Local variables
        INTEGER slen, idx
        CHARACTER*256 warnmsg

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_SET_ENGINE: Invalid windowid value'
        ENDIF

* The window will not exist at the time this is called.
* So just set the values to use when the window is created.

* Get the actual string length of new engine name
        slen = TM_LENSTR(newname)

* An empty engine name uses the default engine
        IF ( slen .EQ. 0 ) THEN
            enginename(windowid) = defaultenginename
            antialias(windowid) = .TRUE.
            thickfactor(windowid) = 1.0
            sts = FERR_OK
            RETURN
        ENDIF

* INDEX is used to match incomplete new engine names
        idx = INDEX('Cairo', newname(1:slen))
        IF ( idx .NE. 1 ) THEN
            idx = INDEX('CAIRO', newname(1:slen))
        ENDIF
        IF ( idx .EQ. 1 ) THEN
            enginename(windowid) = 'Cairo'
            antialias(windowid) = .TRUE.
            sts = FERR_OK
            RETURN
        ENDIF

* If a no-display command-line option was used (and thus
* defaultenginename .EQ. 'Cairo'), do not allow other 
* engines to be used unless forceit is .TRUE.
        IF ( (defaultenginename .EQ. 'Cairo') .AND. .NOT. forceit ) THEN
            warnmsg = '/QUALITY was ignored in SET WINDOW ' //
     .          'since a no-display command-line option was given'
            CALL WARN(warnmsg)
            enginename(windowid) = defaultenginename
            antialias(windowid) = .TRUE.
            sts = FERR_OK
            RETURN
        ENDIF

        idx = INDEX('PipedViewerPQ', newname(1:slen))
        IF ( idx .NE. 1 ) THEN
            idx = INDEX('PIPEDVIEWERPQ', newname(1:slen))
        ENDIF
        IF ( idx .EQ. 1 ) THEN
            enginename(windowid) = 'PipedViewerPQ'
            antialias(windowid) = .TRUE.
            sts = FERR_OK
            RETURN
        ENDIF

        idx = INDEX('PipedImager', newname(1:slen))
        IF ( idx .NE. 1 ) THEN
            idx = INDEX('PIPEDIMAGER', newname(1:slen))
        ENDIF
        IF ( idx .EQ. 1 ) THEN
            enginename(windowid) = 'PipedImager'
            antialias(windowid) = .FALSE.
            sts = FERR_OK
            RETURN
        ENDIF

        IF ( slen .LE. 64 ) THEN
            warnmsg = 'Non-standard graphics engine "'
     .                // newname(1:slen) // '"'
            CALL WARN(warnmsg)
            enginename(windowid) = newname
            antialias(windowid) = .TRUE.
            sts = FERR_OK
            RETURN
        ENDIF

        sts = FERR_INVALID_COMMAND
        RETURN
        END

