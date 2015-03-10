C
C Draws text in a window.
C
C At this time, a completely canned routine to be generalized as needed.
C
        SUBROUTINE FERDRAWTEXT(success)

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

        REAL*4 redfrac, greenfrac, bluefrac, opaqfrac
        REAL*8 mycolor

        CHARACTER*64 mytext
        INTEGER mytextlen
        REAL*4 startx, starty, rotate

C       Draw to the currently active window -
C       probably just leave this as-is
        mywindowid = ACTIVEWINDOW
C       Some sanity checks on the window ID and window
        IF ( (mywindowid .LT. 1) .OR. 
     .       (mywindowid .GT. MAXWINDOWOBJS) ) THEN
            STOP 'FERDRAWTEXT: not an valid window ID'
        ENDIF
        mywindow = WINDOWOBJS(mywindowid)
        IF ( mywindow .EQ. NULLOBJ ) THEN
            STOP 'FERDRAWTEXT: not a valid window'
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

C       Create a color (opaque red) for the text - 
C       or could use one of the standard colors for the window; 
C       e.g., colorobjs(2,activewindow)
        redfrac = 1.0
        greenfrac = 0.0
        bluefrac = 0.0
        opaqfrac = 1.0
        CALL FGDCOLOR(mycolor, mywindow, 
     .                redfrac, greenfrac, bluefrac, opaqfrac)
        IF ( mycolor .EQ. NULLOBJ ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
            CALL FGDFONTDEL(success, myfont)
            success = 0
            RETURN
        ENDIF
 
C       Text to draw
        mytext = "PyFerret!"
        mytextlen = TM_LENSTR(mytext)
C       Position in thousand's of an inch from bottom left
        startx = 4500.0
        starty = 7500.0
C       Rotate in degrees clockwise from horizontal
        rotate = 0.0
C       Draw the text
        CALL FGDDRAWTEXT(success, mywindow, mytext, mytextlen, 
     .                   startx, starty, myfont, mycolor, rotate)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
            CALL FGDCOLORDEL(success, myfont)
            CALL FGDFONTDEL(success, myfont)
            success = 0
            RETURN
        ENDIF
 
C       When done with the color, if not a standard color, delete it; 
C       should always succeed if given a valid color
        CALL FGDCOLORDEL(success, mycolor)

C       When done with the font, if not a standard font, delete it; 
C       should always succeed if given a valid font
        CALL FGDFONTDEL(success, myfont)

C       When all plotting is done, update the window (using the window ID) -
C       this should be eliminated as updates should already be done elsewhere
        CALL FGD_GUWK(mywindowid)

        success = 1
        RETURN
        END
