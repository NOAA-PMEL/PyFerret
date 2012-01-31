*
* Set the name of the graphics engine for windows created
* after this call.  If the engine name is not recognized
* this call does nothing except assign sts.
*
* Input Arguments:
*     newname - name (or partial name) of the engine in
*               uppercase; one of CAIRO, PYQTVIEWER, or
*               PYQTCAIRO (compared in that order)
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

*       Local variables
        INTEGER idx

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

        sts = FERR_INVALID_COMMAND
        RETURN
        END

