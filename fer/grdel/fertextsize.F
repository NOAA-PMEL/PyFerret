C
C Returns the size of given text if drawn with a given font.
C The width is such that continuing text should be positioned 
C at the text start plus this width.  The height will always 
C be the ascent plus descent for the font and is independent 
C of the text.
C
C At this time, a completely canned routine to be generalized as needed.
C
        SUBROUTINE FERTEXTSIZE(success)

        IMPLICIT NONE
        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER success
        INTEGER TM_LENSTR

        CHARACTER*256 errmsg
        INTEGER errmsglen

        INTEGER mywindowid
        REAL*8 mywindow

        CHARACTER*64 fontname
        INTEGER fontnamelen
        REAL*4 fontsize
        INTEGER italicint, boldint, underlineint
        REAL*8 myfont

        CHARACTER*64 mytext
        INTEGER mytextlen
        REAL*4 retwidth, retheight

C       Use the currently active window -
C       probably just leave this as-is
        mywindowid = ACTIVEWINDOW
C       Some sanity checks on the window ID and window
        IF ( (mywindowid .LT. 1) .OR. 
     .       (mywindowid .GT. MAXWINDOWOBJS) ) THEN
            STOP 'FERTEXTSIZE: not an valid window ID'
        ENDIF
        mywindow = WINDOWOBJS(mywindowid)
        IF ( mywindow .EQ. NULLOBJ ) THEN
            STOP 'FERTEXTSIZE: not a valid window'
        ENDIF

C       Create a font (Courier 24pt bold italic) to use for the string -
C       probably store at least one standard font for reuse for each window
        fontname = 'Courier'
        fontnamelen = TM_LENSTR(fontname)
        fontsize = 24.0
C       Italic? - zero = no, non-zero = yes
        italicint = 1
C       Bold? - zero = no, non-zero = yes
        boldint = 1
C       Underlined? - zero = no, non-zero = yes
C       Not sure if underline support is provided by Cairo
        underlineint = 0
        CALL FGDFONT(myfont, mywindow, fontname, fontnamelen, 
     .               fontsize, italicint, boldint, underlineint)
        IF ( myfont .EQ. NULLOBJ ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
            success = 0
            RETURN
        ENDIF

C       Text to use
        mytext = "PyFerret!"
        mytextlen = TM_LENSTR(mytext)
C       Measure the text
        CALL FGDTEXTSIZE(success, mywindow, mytext, mytextlen, 
     .                   myfont, retwidth, retheight)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
            CALL FGDFONTDEL(success, myfont)
            success = 0
            RETURN
        ENDIF
 
C       Reporting the text size to err_lun to validate
        WRITE(errmsg, 100) retwidth, retheight
  100   FORMAT('Text width = ',F8.1,', height = ',F8.1)
        errmsglen = TM_LENSTR(errmsg)
        CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)

C       When done with the font, if not a standard font, delete it; 
C       should always succeed if given a valid font
        CALL FGDFONTDEL(success, myfont)

        success = 1
        RETURN
        END
