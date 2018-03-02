*
* Set the name of the graphics engine for windows created
* after this call. 
*
* Input Arguments:
*     windowid - window number to use
*     newname - name (or partial name) of the engine;
*               standard recognized names are Cairo, 
*               CAIRO, PipedViewerPQ, PIPEDVIEWERPQ, 
*               PipedImager, PIPEDIMAGER, NoDisplayPQ,
*               an NODISPLAYPQ
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
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'errmsg.parm'
        INCLUDE 'plot_setup.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

*       Function declarations
        INTEGER TM_LENSTR

*       Local variables
        INTEGER slen, idx, errstrlen
        CHARACTER*256 errstr

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_SET_ENGINE: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            sts = FERR_INVALID_COMMAND
            RETURN
        ENDIF

* The window will not exist at the time this is called.
* So just set the values to use when the window is created.

* Get the actual string length of new engine name
        slen = TM_LENSTR(newname)

* An empty engine name uses the default engine
        IF ( slen .EQ. 0 ) THEN
            enginename(windowid) = defaultenginename
            antialias(windowid) = .TRUE.
            thickfactor(windowid) = dflt_thicken
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
            errstr = '/QUALITY was ignored in SET WINDOW ' //
     .          'since a no-display command-line option was given'
            CALL WARN(errstr)
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

        idx = INDEX('NoDisplayPQ', newname(1:slen))
        IF ( idx .NE. 1 ) THEN
            idx = INDEX('NODISPLAYPQ', newname(1:slen))
        ENDIF
        IF ( idx .EQ. 1 ) THEN
            enginename(windowid) = 'NoDisplayPQ'
            antialias(windowid) = .TRUE.
            sts = FERR_OK
            RETURN
        ENDIF

        IF ( slen .LE. 64 ) THEN
            errstr = 'Non-standard graphics engine "'
     .                // newname(1:slen) // '"'
            CALL WARN(errstr)
            enginename(windowid) = newname
            antialias(windowid) = .TRUE.
            sts = FERR_OK
            RETURN
        ENDIF

        sts = FERR_INVALID_COMMAND
        RETURN
        END

