*
* Definition of GQWKC
* (inquire workstatrion connection and type)
*

        SUBROUTINE FGD_GQWKC(windowid, istat, connectid, wstype)

        IMPLICIT NONE
        INTEGER windowid, istat, connectid, wstype

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

*       Sanity check
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GQWKC: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            istat = 1
            RETURN
        ENDIF

*       Canned return values based of XWindows display
        connectid = 1
        wstype = 4
        istat = 0

        RETURN
        END
