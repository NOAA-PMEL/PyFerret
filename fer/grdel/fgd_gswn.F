*
* Set the user (world) coordinates for a view transformation
*

        SUBROUTINE FGD_GSWN(transnum, xmin, xmax, ymin, ymax)

        IMPLICIT NONE
        INTEGER transnum
        REAL*4 xmin, xmax, ymin, ymax

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

*       Sanity checks - transform 0 is implicit identity
        IF ( (transnum .LT. 1) .OR.
     .       (transnum .GT. maxtransforms) ) THEN
            errstr = 'FGD_GSWN: invalid transnum'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (xmax - xmin) .LE. 0.0 ) THEN
            errstr = 'FGD_GSWN: invalid xmin, xmax values'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            WRITE(errstr,*) 'FGD_GSWN: xmin = ', xmin, ', xmax = ', xmax
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (ymax - ymin) .LE. 0.0 ) THEN
            errstr = 'FGD_GSWN: invalid ymin, ymax values'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            WRITE(errstr,*) 'FGD_GSWN: ymin = ', ymin, ', ymax = ', ymax
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        viewleftcoord(transnum) = xmin
        viewrightcoord(transnum) = xmax
        viewbottomcoord(transnum) = ymin
        viewtopcoord(transnum) = ymax

        RETURN
        END

