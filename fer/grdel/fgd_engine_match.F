*
* Compare the graphics engine name given to that used
* for indicated window.
*
* Input Arguments:
*     windowid - window number to use
*     engname - name (or partial name) of the engine;
*               standard recognized names are Cairo, 
*               PyQtViewer, or PyQtCairo (compared
*               in that order, so PyQt match PyQtViewer)
* Returns
*     .TRUE. if the window uses the indicated engine; 
*     otherwise .FALSE.
*
        LOGICAL FUNCTION FGD_ENGINE_MATCH(windowid, engname)

        IMPLICIT NONE
        CHARACTER*(*) engname
        INTEGER windowid

        INCLUDE 'ferret.parm'
        INCLUDE 'errmsg.parm'
        INCLUDE 'fgrdel.cmn'

*       Function declarations
        INTEGER TM_LENSTR

*       Local variables
        INTEGER slen, idx

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_ENGINE_MATCH: Invalid windowid value'
        ENDIF

* Get the actual string length of given engine name
        slen = TM_LENSTR(engname)

* An empty engine name uses the default engine
        IF ( slen .EQ. 0 ) THEN
            FGD_ENGINE_MATCH = 
     .          enginename(windowid) .EQ. defaultenginename
            RETURN
        ENDIF

* INDEX is used to match incomplete new engine names
        idx = INDEX('Cairo', engname(1:slen))
        IF ( idx .EQ. 1 ) THEN
            FGD_ENGINE_MATCH = enginename(windowid) .EQ. 'Cairo'
            RETURN
        ENDIF

* Note that 'PyQt' will give 'PyQtViewer' and not 'PyQtCairo'
        idx = INDEX('PyQtViewer', engname(1:slen))
        IF ( idx .EQ. 1 ) THEN
            FGD_ENGINE_MATCH = enginename(windowid) .EQ. 'PyQtViewer'
            RETURN
        ENDIF

        idx = INDEX('PyQtCairo', engname(1:slen))
        IF ( idx .EQ. 1 ) THEN
            FGD_ENGINE_MATCH = enginename(windowid) .EQ. 'PyQtCairo'
            RETURN
        ENDIF

        IF ( slen .LE. 64 ) THEN
            FGD_ENGINE_MATCH = enginename(windowid) .EQ. engname
            RETURN
        ENDIF

        FGD_ENGINE_MATCH = .FALSE.
        RETURN
        END

