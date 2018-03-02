*
* Define the normalized device coordinates limits for the full window.
* (set workstation window)
* In other words: the full window, instead of
* going from zero to one for X and Y, 
* it goes from xmin to xmax for X 
*          and ymin to ymax for Y
* Ferret always sets xmin and ymin to zero,
* and either xmax or ymax to one, with the other to (0.0,1.0],
* thus used to adjust for the aspect ratio of the monitor.
*

        SUBROUTINE FGD_GSWKWN(windowid, xmin, xmax, ymin, ymax)

        IMPLICIT NONE
        INTEGER windowid
        REAL*4 xmin, xmax, ymin, ymax

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSWKWN: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( xmin .NE. 0.0 ) THEN
            errstr = 'FGD_GSWKWN: unexpected non-zero xmin'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( ymin .NE. 0.0 ) THEN
            errstr = 'FGD_GSWKWN: unexpected non-zero ymin'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( xmax .GT. 1.0 ) THEN
            errstr = 'FGD_GSWKWN: unexpected xmax greater than one'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( ymax .GT. 1.0 ) THEN
            errstr = 'FGD_GSWKWN: unexpected ymax greater than one'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        ndcxmax(windowid) = xmax
        ndcymax(windowid) = ymax

        RETURN
        END

