*
* Set the name of the graphics engine for windows created
* after this call. 
*
* Input Arguments:
*     newname - name (or partial name) of the engine in
*               uppercase; standard recognized names are
*               CAIRO, PYQTVIEWER, or PYQTCAIRO (compared
*               in that order)
* Output Arguments:
*     sts - FERR_OK if successful; 
*           FERR_INVALID_COMMAND otherwise
*
        SUBROUTINE FGD_SET_ENGINE(newname, sts)

        IMPLICIT NONE
        CHARACTER*(*) newname
        INTEGER sts

        INCLUDE 'ferret.parm'
        INCLUDE 'errmsg.parm'
        INCLUDE 'fgrdel.cmn'

*       Function declarations
        INTEGER TM_LENSTR

*       Local variables
        INTEGER idx
        CHARACTER*256 warnmsg

        idx = INDEX("CAIRO", newname)
        IF ( idx .EQ. 1 ) THEN
            enginename = "Cairo"
            sts = FERR_OK
            RETURN
        ENDIF

        idx = INDEX("PYQTVIEWER", newname)
        IF ( idx .EQ. 1 ) THEN
            enginename = "PyQtViewer"
            sts = FERR_OK
            RETURN
        ENDIF

        idx = INDEX("PYQTCAIRO", newname)
        IF ( idx .EQ. 1 ) THEN
            enginename = "PyQtCairo"
            sts = FERR_OK
            RETURN
        ENDIF

        idx = TM_LENSTR(newname)
        IF ( (idx .GT. 1) .AND. (idx .LT. 65) ) THEN
            CALL WARN('Non-standard graphics engine "'
     .                // newname(1:idx) // '"')
            enginename = newname
            sts = FERR_OK
            RETURN
        ENDIF

        sts = FERR_INVALID_COMMAND
        RETURN
        END

