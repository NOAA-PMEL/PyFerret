C
C Draws text in a window.
C
C  fontsize	In user coordinates
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


        SUBROUTINE FERDRAWTEXT(text, fontname, fontsize, 
     .    redfrac, greenfrac, bluefrac, opaqfrac,
     .    startx, starty, rotate, 
     .    italic, bold, under, success)

        IMPLICIT NONE
        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        CHARACTER*(*) text, fontname
        REAL*4 fontsize
        REAL*4 redfrac, greenfrac, bluefrac, opaqfrac
        REAL*4 startx, starty, rotate
        INTEGER italic, bold, under
        INTEGER success

        INTEGER TM_LENSTR

        CHARACTER*2048 errstr
        INTEGER errstrlen
        INTEGER mywindowid
        REAL*8 mywindow
        INTEGER fontnamelen
        REAL*8 myfont
        REAL*8 mycolor
        INTEGER textlen

C       Draw to the currently active window -
        mywindowid = ACTIVEWINDOW
C       Some sanity checks on the window ID and window
        IF ( (mywindowid .LT. 1) .OR. 
     .       (mywindowid .GT. MAXWINDOWOBJS) ) THEN
            errstr = 'FERDRAWTEXT: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            success = 0
            RETURN
        ENDIF
        mywindow = WINDOWOBJS(mywindowid)
        IF ( mywindow .EQ. NULLOBJ ) THEN
            errstr = 'FERDRAWTEXT: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            success = 0
            RETURN
        ENDIF

C       Create the font to use for the string -
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

C       Create a color for the text with red, green, blue fractions - 
C       or could use one of the standard colors for the window; 
C       e.g., colorobjs(2,activewindow)

        CALL FGDCOLOR(mycolor, mywindow, 
     .                redfrac, greenfrac, bluefrac, opaqfrac)
        IF ( mycolor .EQ. NULLOBJ ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            CALL FGDFONTDEL(success, myfont)
            success = 0
            RETURN
        ENDIF
 
C       Draw the text
        textlen = TM_LENSTR(text)
        CALL FGDDRAWTEXT(success, mywindow, text, textlen, 
     .                   startx, starty, myfont, mycolor, rotate)
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            CALL FGDCOLORDEL(success, myfont)
            CALL FGDFONTDEL(success, myfont)
            success = 0
            RETURN
        ENDIF
 
C       Delete the color created above
        CALL FGDCOLORDEL(success, mycolor)

C       Delete the font created above
        CALL FGDFONTDEL(success, myfont)

        success = 1
        RETURN
        END
