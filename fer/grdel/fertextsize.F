C
C Returns the size of given text if drawn with a given font.
C The width is such that continuing text should be positioned 
C at the text start plus this width.  The height will always 
C be the ascent plus descent for the font and is independent 
C of the text.  The fontsize is in user coordinates.
C
C
        SUBROUTINE FERTEXTSIZE(text, fontname, fontsize, 
     .          italic, bold, under, retwidth, retheight, success)

        IMPLICIT NONE
        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        CHARACTER*(*) text, fontname
        REAL*4 fontsize
        INTEGER italic, bold, under
        REAL*4 retwidth, retheight
        INTEGER success

        INTEGER TM_LENSTR

        CHARACTER*2048 errstr
        INTEGER errstrlen
        INTEGER mywindowid
        REAL*8 mywindow
        INTEGER fontnamelen
        REAL*8 myfont
        INTEGER textlen

C       Use the currently active window -
C       probably just leave this as-is
        mywindowid = ACTIVEWINDOW
C       Some sanity checks on the window ID and window
        IF ( (mywindowid .LT. 1) .OR. 
     .       (mywindowid .GT. MAXWINDOWOBJS) ) THEN
            errstr = 'FERTEXTSIZE: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            success = 0
            RETURN
        ENDIF
        mywindow = WINDOWOBJS(mywindowid)
        IF ( mywindow .EQ. NULLOBJ ) THEN
            errstr = 'FERTEXTSIZE: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            success = 0
            RETURN
        ENDIF

C       Create a font to use for the string -
C       No underline support is provided by Cairo,
C       so under is ignored at this time
        fontnamelen = TM_LENSTR(fontname)
        CALL FGDFONT(myfont, mywindow, fontname, fontnamelen, 
     .               fontsize, italic, bold, under)
        IF ( myfont .EQ. NULLOBJ ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            success = 0
            RETURN
        ENDIF

C       Text to use
        textlen = TM_LENSTR(text)
C       Measure the text
        CALL FGDTEXTSIZE(success, mywindow, text, textlen, 
     .                   myfont, retwidth, retheight)
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            CALL FGDFONTDEL(success, myfont)
            success = 0
            RETURN
        ENDIF
 
C       Delete the font created above
        CALL FGDFONTDEL(success, myfont)

        success = 1
        RETURN
        END
