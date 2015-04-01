C
C Returns the size of given text if drawn with a given font.
C The width is such that continuing text should be positioned 
C at the text start plus this width.  The height will always 
C be the ascent plus descent for the font and is independent 
C of the text.
C
C
        SUBROUTINE FERTEXTSIZE(in_text, infont_name, fontsize, 
     .          italic, bold, under, retwidth, retheight, success)

        IMPLICIT NONE
        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'	

        REAL*4 fontsize
	REAL*4 retwidth, retheight
	INTEGER italic, bold, under

	INTEGER success

	CHARACTER*(*) in_text, infont_name

        INTEGER TM_LENSTR

        CHARACTER*256 errmsg
        INTEGER errmsglen

        INTEGER mywindowid
        REAL*8 mywindow

        CHARACTER*64 fontname
        INTEGER fontnamelen
        REAL*8 myfont

        CHARACTER*64 mytext
        INTEGER mytextlen

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
        fontname = infont_name
        fontnamelen = TM_LENSTR(fontname)

C       Not sure if underline support is provided by Cairo
        under = 0
        CALL FGDFONT(myfont, mywindow, fontname, fontnamelen, 
     .               fontsize, italic, bold, under)
        IF ( myfont .EQ. NULLOBJ ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
            success = 0
            RETURN
        ENDIF

C       Text to use
        mytext = in_text
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
c        WRITE(errmsg, 100) retwidth, retheight
c  100   FORMAT('Text width = ',F8.1,', height = ',F8.1)
c        errmsglen = TM_LENSTR(errmsg)
c        CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)

C       When done with the font, if not a standard font, delete it; 
C       should always succeed if given a valid font
        CALL FGDFONTDEL(success, myfont)

        success = 1
        RETURN
        END
