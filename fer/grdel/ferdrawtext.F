C
C Draws text in a window.
C
C  fontsize	In points
C  redfrac
C  greenfrac	RGB in percent
C  bluefrac
C  opaqfrac	opacity in percent
C  startx	position in thousandths of an inch from bottom left
C  starty
C  rotate	rotation in degrees clockwise from horizontal
C  italic 
C  bold	lettering linestyle flags: 0=false, 1=true
C  under


        SUBROUTINE FERDRAWTEXT(in_text, infont_name, fontsize, 
     .    redfrac, greenfrac, bluefrac, opaqfrac,
     .    startx, starty, rotate, 
     .    italic, bold, under, success)

        IMPLICIT NONE
        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'


        REAL*4 fontsize
        REAL*4 redfrac, greenfrac, bluefrac, opaqfrac
        REAL*4 startx, starty, rotate
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
        REAL*8 mycolor

        CHARACTER*64 mytext
        INTEGER mytextlen

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
        fontname = infont_name
        fontnamelen = TM_LENSTR(fontname)

        CALL FGDFONT(myfont, mywindow, fontname, fontnamelen, 
     .               fontsize, italic, bold, under)
        IF ( myfont .EQ. NULLOBJ ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
            success = 0
            RETURN
        ENDIF

C       Create a color for the text with red, green, blue fractions - 
C       or could use one of the standard colors for the window; 
C       e.g., colorobjs(2,activewindow)

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
        mytext = in_text
        mytextlen = TM_LENSTR(mytext)

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
