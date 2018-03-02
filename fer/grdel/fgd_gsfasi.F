*
* Set the hatch style to use when creating a temporary brush.
* This unsets the active brush.
* (set fill area style index)
*

        SUBROUTINE FGD_GSFASI(hstyle)

        IMPLICIT NONE
        INTEGER hstyle

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

        IF ( hstyle .EQ. 1 ) THEN
            lastbrushstyle = "hor"
            lastbstylen = 3
        ELSE IF ( hstyle .EQ. 2 ) THEN
            lastbrushstyle = "fdiag"
            lastbstylen = 5
        ELSE IF ( hstyle .EQ. 3 ) THEN
            lastbrushstyle = "ver"
            lastbstylen = 3
        ELSE IF ( hstyle .EQ. 4 ) THEN
            lastbrushstyle = "bdiag"
            lastbstylen = 5
        ELSE IF ( hstyle .EQ. 5 ) THEN
            lastbrushstyle = "cross"
            lastbstylen = 5
        ELSE IF ( hstyle .EQ. 6 ) THEN
            lastbrushstyle = "diagcross"
            lastbstylen = 9
        ELSE
            errstr = 'FGD_GSFASI: invalid hstyle'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        activebrush = 0

        RETURN
        END

