*
* Set the fill style to use when creating a temporary brush.
* This unsets the active brush.
* (set fill area interior style)
*

        SUBROUTINE FGD_GSFAIS(style)

        IMPLICIT NONE
        INTEGER style

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

        IF ( style .EQ. 1 ) THEN
*           Solid brush
            lastbrushstyle = 'solid'
            lastbstylen = 5
        ELSE IF ( style .EQ. 3 ) THEN
*           Hatch brush - default to diagcross
            lastbrushstyle = 'diagcross'
            lastbstylen = 9
        ELSE
            errstr = 'FGD_GSFAIS: invalid style'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        activebrush = 0

        RETURN
        END

