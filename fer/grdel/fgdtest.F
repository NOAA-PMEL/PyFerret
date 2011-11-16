      SUBROUTINE FGDTEST(mywindow)
      IMPLICIT NONE

!     The following are actually pointers to objects
!     The window (bindings object) created by this
!     subroutine is returned in the argument mywindow.
      REAL*8 mywindow, myfont, mycolors(17), mybrush, mypen, mysymbol,
     .       multicolors(6)
      EQUIVALENCE (mycolors(11), multicolors(1))

!     The following are actual the variables they claim to be
      CHARACTER*2500 errmsg
      CHARACTER*64   viewername, wintitle, fontfamily, style,
     .               capstyle, joinstyle, text, symbol
      REAL*4         pentaptsx(5), pentaptsy(5), ptsx(6), ptsy(6)
      INTEGER*4      errmsglen, success, k

!     Create the window
      viewername = 'PyQtPipedViewer'
      wintitle = 'Tester'
      CALL FGDWINCREATE(mywindow, viewername, 15, wintitle, 6,
     .                  5000.0, 5000.0, 1)
      IF ( mywindow .EQ. 0.0 ) THEN
          WRITE(*,*) '1 FGDWINDOW returned a NULL window'
          GOTO 900
      ENDIF

!     Create the one font that will be used here
!     - default font, 1/5th of the view in size
      fontfamily = ' '
      CALL FGDFONT(myfont, mywindow, fontfamily, 0, 200.0, 0, 0, 0)
      IF ( myfont .EQ. 0.0 ) THEN
          WRITE(*,*) '2 FGDFONT returned a NULL font'
          GOTO 900
      ENDIF

!     Create a list of colors that will be used here
!     Opaque Black
      CALL FGDCOLOR(mycolors(1), mywindow, 0.0, 0.0, 0.0, 1.0)
      IF ( mycolors(1) .EQ. 0.0) THEN
          WRITE(*,*) '3 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Opaque White
      CALL FGDCOLOR(mycolors(2), mywindow, 1.0, 1.0, 1.0, 1.0)
      IF ( mycolors(2) .EQ. 0.0) THEN
          WRITE(*,*) '4 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Opaque Red
      CALL FGDCOLOR(mycolors(3), mywindow, 1.0, 0.0, 0.0, 1.0)
      IF ( mycolors(3) .EQ. 0.0) THEN
          WRITE(*,*) '5 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Opaque Yellowish
      CALL FGDCOLOR(mycolors(4), mywindow, 0.6, 0.5, 0.0, 1.0)
      IF ( mycolors(4) .EQ. 0.0) THEN
          WRITE(*,*) '6 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Opaque Green
      CALL FGDCOLOR(mycolors(5), mywindow, 0.0, 1.0, 0.0, 1.0)
      IF ( mycolors(5) .EQ. 0.0) THEN
          WRITE(*,*) '7 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Opaque Cyan
      CALL FGDCOLOR(mycolors(6), mywindow, 0.0, 0.5, 0.5, 1.0)
      IF ( mycolors(6) .EQ. 0.0) THEN
          WRITE(*,*) '8 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Opaque Blue
      CALL FGDCOLOR(mycolors(7), mywindow, 0.0, 0.0, 1.0, 1.0)
      IF ( mycolors(7) .EQ. 0.0) THEN
          WRITE(*,*) '9 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Opaque Magenta
      CALL FGDCOLOR(mycolors(8), mywindow, 0.5, 0.0, 0.5, 1.0)
      IF ( mycolors(8) .EQ. 0.0) THEN
          WRITE(*,*) '10 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Translucent Black
      CALL FGDCOLOR(mycolors(9), mywindow, 0.0, 0.0, 0.0, 0.25)
      IF ( mycolors(9) .EQ. 0.0) THEN
          WRITE(*,*) '11 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Translucent White
      CALL FGDCOLOR(mycolors(10), mywindow, 1.0, 1.0, 1.0, 0.25)
      IF ( mycolors(10) .EQ. 0.0) THEN
          WRITE(*,*) '12 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Translucent Red
      CALL FGDCOLOR(mycolors(11), mywindow, 1.0, 0.0, 0.0, 0.25)
      IF ( mycolors(11) .EQ. 0.0) THEN
          WRITE(*,*) '13 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Translucent Yellowish
      CALL FGDCOLOR(mycolors(12), mywindow, 0.6, 0.5, 0.0, 0.25)
      IF ( mycolors(12) .EQ. 0.0) THEN
          WRITE(*,*) '14 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Translucent Green
      CALL FGDCOLOR(mycolors(13), mywindow, 0.0, 1.0, 0.0, 0.25)
      IF ( mycolors(13) .EQ. 0.0) THEN
          WRITE(*,*) '15 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Translucent Cyan
      CALL FGDCOLOR(mycolors(14), mywindow, 0.0, 0.5, 0.5, 0.25)
      IF ( mycolors(14) .EQ. 0.0) THEN
          WRITE(*,*) '16 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Translucent Blue
      CALL FGDCOLOR(mycolors(15), mywindow, 0.0, 0.0, 1.0, 0.25)
      IF ( mycolors(15) .EQ. 0.0) THEN
          WRITE(*,*) '17 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Translucent Magenta
      CALL FGDCOLOR(mycolors(16), mywindow, 0.5, 0.0, 0.5, 0.25)
      IF ( mycolors(16) .EQ. 0.0) THEN
          WRITE(*,*) '18 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Transparent White
      CALL FGDCOLOR(mycolors(17), mywindow, 1.0, 1.0, 1.0, 0.0)
      IF ( mycolors(17) .EQ. 0.0) THEN
          WRITE(*,*) '19 FGDCOLOR returned a NULL color'
          GOTO 900
      ENDIF
!     Clear the window in transparent white
      CALL FGDWINCLEAR(success, mywindow, mycolors(17))
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '20 FGDWINDCLEAR failed'
          GOTO 900
      ENDIF
!     Create a view in the top left corner
      CALL FGDVIEWBEGIN(success, mywindow,
     .                  0.0, 0.5, 0.5, 1.0,
     .                  0.0, 0.0, 1000.0, 1000.0, 1)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '21 FGDVIEWBEGIN failed'
          GOTO 900
      ENDIF
!     Draw a translucent black rentagle over most of the view
      style = 'solid'
      CALL FGDBRUSH(mybrush, mywindow, mycolors(9), style, 5)
      IF ( mybrush .EQ. 0.0 ) THEN
          WRITE(*,*) '22 FGDBRUSH returned a NULL brush'
          GOTO 900
      ENDIF
!     Do not use a pen
      mypen = 0.0
      CALL FGDDRAWRECT(success, mywindow, 50.0, 50.0,
     .                 950.0, 950.0, mybrush, mypen)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '23 FGDDRAWRECT failed'
          GOTO 900
      ENDIF
      CALL FGDBRUSHDEL(success, mybrush)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '24 FGDBRUSHDEL failed'
          GOTO 900
      ENDIF
!     x and y coordinates of the vertices of a pentagon
!     (roughly) centered in a 1000 x 1000 square
      pentaptsx(1) = 504.5
      pentaptsx(2) = 100.0
      pentaptsx(3) = 254.5
      pentaptsx(4) = 754.5
      pentaptsx(5) = 909.0
      pentaptsy(1) = 100.0
      pentaptsy(2) = 393.9
      pentaptsy(3) = 869.4
      pentaptsy(4) = 869.4
      pentaptsy(5) = 393.9
!     Draw a opaque blue polygon with solid black outline
      style = 'solid'
      CALL FGDBRUSH(mybrush, mywindow, mycolors(7), style, 5)
      IF ( mybrush .EQ. 0.0 ) THEN
          WRITE(*,*) '25 FGDBRUSH returned a NULL brush'
          GOTO 900
      ENDIF
      style = 'solid'
      capstyle = 'round'
      joinstyle = 'round'
      CALL FGDPEN(mypen, mywindow, mycolors(1), 25.0,
     .            style, 5, capstyle, 5, joinstyle, 5)
      IF ( mypen .EQ. 0.0 ) THEN
          WRITE(*,*) '26 FGDPEN returned a NULL pen'
          GOTO 900
      ENDIF
      CALL FGDDRAWPOLYGON(success, mywindow, pentaptsx, pentaptsy,
     .                    5, mybrush, mypen)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '27 FGDDRAWPOLYGON failed'
          GOTO 900
      ENDIF
      CALL FGDPENDEL(success, mypen)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '28 FGDPENDEL failed'
          GOTO 900
      ENDIF
      CALL FGDBRUSHDEL(success, mybrush)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '29 FGDBRUSHDEL failed'
          GOTO 900
      ENDIF
!     Draw some red text strings
      text = 'y=100'
      CALL FGDDRAWTEXT(success, mywindow, text, 5, 100.0, 100.0,
     .                 myfont, mycolors(3), 0.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '30 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
      text = 'y=300'
      CALL FGDDRAWTEXT(success, mywindow, text, 5, 100.0, 300.0,
     .                 myfont, mycolors(3), 0.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '31 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
      text = 'y=500'
      CALL FGDDRAWTEXT(success, mywindow, text, 5, 100.0, 500.0,
     .                 myfont, mycolors(3), 0.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '32 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
      text = 'y=700'
      CALL FGDDRAWTEXT(success, mywindow, text, 5, 100.0, 700.0,
     .                 myfont, mycolors(3), 0.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '33 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
!     End of this view
      CALL FGDVIEWEND(success, mywindow)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '34 FGDVIEWEND failed'
          GOTO 900
      ENDIF
!     Window should be visible, but just be sure
      CALL FGDWINSETVIS(success, mywindow, 1)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '35 FGDWINSETVIS failed'
          GOTO 900
      ENDIF
!     Create a view of almost the whole window
      CALL FGDVIEWBEGIN(success, mywindow,
     .                  0.05, 0.05, 0.95, 0.95,
     .                  0.0, 0.0, 1000.0, 1000.0, 1)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '36 FGDVIEWBEGIN failed'
          GOTO 900
      ENDIF
!     Draw a translucent multicolor rectangle covering most of the window
      CALL FGDDRAWMULTICOLOR(success, mywindow, 50.0, 50.0,
     .                       950.0, 950.0, 2, 3, multicolors)
       IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '37 FGDDRAWMULTICOLOR failed'
          GOTO 900
      ENDIF
!     Draw letters indicating the expected colors
      text = 'R'
      CALL FGDDRAWTEXT(success, mywindow, text, 1, 200.0, 600.0,
     .                 myfont, mycolors(1), -45.0)
       IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '38 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
      text = 'Y'
      CALL FGDDRAWTEXT(success, mywindow, text, 1, 200.0, 150.0,
     .                 myfont, mycolors(1), -45.0)
       IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '39 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
      text = 'G'
      CALL FGDDRAWTEXT(success, mywindow, text, 1, 500.0, 600.0,
     .                 myfont, mycolors(1), -45.0)
       IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '40 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
      text = 'C'
      CALL FGDDRAWTEXT(success, mywindow, text, 1, 500.0, 150.0,
     .                 myfont, mycolors(1), -45.0)
       IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '41 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
      text = 'B'
      CALL FGDDRAWTEXT(success, mywindow, text, 1, 800.0, 600.0,
     .                 myfont, mycolors(1), -45.0)
       IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '42 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
      text = 'M'
      CALL FGDDRAWTEXT(success, mywindow, text, 1, 800.0, 150.0,
     .                 myfont, mycolors(1), -45.0)
       IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '43 FGDDRAWTEXT failed'
          GOTO 900
      ENDIF
!     End of this view
      CALL FGDVIEWEND(success, mywindow)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '44 FGDVIEWEND failed'
          GOTO 900
      ENDIF
!     Window should be visible, but just be sure
      CALL FGDWINSETVIS(success, mywindow, 1)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '45 FGDWINSETVIS failed'
          GOTO 900
      ENDIF
!     Create a view of the whole window
      CALL FGDVIEWBEGIN(success, mywindow,
     .                  0.0, 0.0, 1.0, 1.0,
     .                  0.0, 0.0, 1000.0, 1000.0, 1)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '46 FGDVIEWBEGIN failed'
          GOTO 900
      ENDIF
!     Draw points using various symbols
      ptsy(1) = 100
      ptsy(2) = 300
      ptsy(3) = 500
      ptsy(4) = 700
      ptsy(5) = 900
      symbol = '.'
      CALL FGDSYMBOL(mysymbol, mywindow, symbol, 1)
      IF ( mysymbol .EQ. 0.0 ) THEN
          WRITE(*,*) '47 FGDSYMBOL returned a NULL symbol'
          GOTO 900
      ENDIF
      ptsx(1) = 100
      ptsx(2) = 100
      ptsx(3) = 100
      ptsx(4) = 100
      ptsx(5) = 100
      CALL FGDDRAWPOINTS(success, mywindow, ptsx, ptsy, 5, mysymbol,
     .                   mycolors(1), 50.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '48 FGDDRAWPOINTS failed'
          GOTO 900
      ENDIF
      CALL FGDSYMBOLDEL(success, mysymbol)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '49 FGDSYMBOLDEL failed'
          GOTO 900
      ENDIF
      symbol = 'o'
      CALL FGDSYMBOL(mysymbol, mywindow, symbol, 1)
      IF ( mysymbol .EQ. 0.0 ) THEN
          WRITE(*,*) '50 FGDSYMBOL returned a NULL symbol'
          GOTO 900
      ENDIF
      ptsx(1) = 200
      ptsx(2) = 200
      ptsx(3) = 200
      ptsx(4) = 200
      ptsx(5) = 200
      CALL FGDDRAWPOINTS(success, mywindow, ptsx, ptsy, 5, mysymbol,
     .                   mycolors(1), 50.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '51 FGDDRAWPOINTS failed'
          GOTO 900
      ENDIF
      CALL FGDSYMBOLDEL(success, mysymbol)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '52 FGDSYMBOLDEL failed'
          GOTO 900
      ENDIF
      symbol = '+'
      CALL FGDSYMBOL(mysymbol, mywindow, symbol, 1)
      IF ( mysymbol .EQ. 0.0 ) THEN
          WRITE(*,*) '53 FGDSYMBOL returned a NULL symbol'
          GOTO 900
      ENDIF
      ptsx(1) = 300
      ptsx(2) = 300
      ptsx(3) = 300
      ptsx(4) = 300
      ptsx(5) = 300
      CALL FGDDRAWPOINTS(success, mywindow, ptsx, ptsy, 5, mysymbol,
     .                   mycolors(7), 50.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '54 FGDDRAWPOINTS failed'
          GOTO 900
      ENDIF
      CALL FGDSYMBOLDEL(success, mysymbol)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '55 FGDSYMBOLDEL failed'
          GOTO 900
      ENDIF
      symbol = 'x'
      CALL FGDSYMBOL(mysymbol, mywindow, symbol, 1)
      IF ( mysymbol .EQ. 0.0 ) THEN
          WRITE(*,*) '56 FGDSYMBOL returned a NULL symbol'
          GOTO 900
      ENDIF
      ptsx(1) = 400
      ptsx(2) = 400
      ptsx(3) = 400
      ptsx(4) = 400
      ptsx(5) = 400
      CALL FGDDRAWPOINTS(success, mywindow, ptsx, ptsy, 5, mysymbol,
     .                   mycolors(1), 50.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '57 FGDDRAWPOINTS failed'
          GOTO 900
      ENDIF
      CALL FGDSYMBOLDEL(success, mysymbol)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '58 FGDSYMBOLDEL failed'
          GOTO 900
      ENDIF
      symbol = '*'
      CALL FGDSYMBOL(mysymbol, mywindow, symbol, 1)
      IF ( mysymbol .EQ. 0.0 ) THEN
          WRITE(*,*) '59 FGDSYMBOL returned a NULL symbol'
          GOTO 900
      ENDIF
      ptsx(1) = 500
      ptsx(2) = 500
      ptsx(3) = 500
      ptsx(4) = 500
      ptsx(5) = 500
      CALL FGDDRAWPOINTS(success, mywindow, ptsx, ptsy, 5, mysymbol,
     .                   mycolors(1), 50.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '60 FGDDRAWPOINTS failed'
          GOTO 900
      ENDIF
      CALL FGDSYMBOLDEL(success, mysymbol)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '61 FGDSYMBOLDEL failed'
          GOTO 900
      ENDIF
      symbol = '^'
      CALL FGDSYMBOL(mysymbol, mywindow, symbol, 1)
      IF ( mysymbol .EQ. 0.0 ) THEN
          WRITE(*,*) '62 FGDSYMBOL returned a NULL symbol'
          GOTO 900
      ENDIF
      ptsx(1) = 600
      ptsx(2) = 600
      ptsx(3) = 600
      ptsx(4) = 600
      ptsx(5) = 600
      CALL FGDDRAWPOINTS(success, mywindow, ptsx, ptsy, 5, mysymbol,
     .                   mycolors(7), 50.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '63 FGDDRAWPOINTS failed'
          GOTO 900
      ENDIF
      CALL FGDSYMBOLDEL(success, mysymbol)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '64 FGDSYMBOLDEL failed'
          GOTO 900
      ENDIF
      symbol = '#'
      CALL FGDSYMBOL(mysymbol, mywindow, symbol, 1)
      IF ( mysymbol .EQ. 0.0 ) THEN
          WRITE(*,*) '65 FGDSYMBOL returned a NULL symbol'
          GOTO 900
      ENDIF
      ptsx(1) = 700
      ptsx(2) = 700
      ptsx(3) = 700
      ptsx(4) = 700
      ptsx(5) = 700
      CALL FGDDRAWPOINTS(success, mywindow, ptsx, ptsy, 5, mysymbol,
     .                   mycolors(1), 50.0)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '66 FGDDRAWPOINTS failed'
          GOTO 900
      ENDIF
      CALL FGDSYMBOLDEL(success, mysymbol)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '67 FGDSYMBOLDEL failed'
          GOTO 900
      ENDIF
      style = 'dash'
      capstyle = 'round'
      joinstyle = 'round'
      CALL FGDPEN(mypen, mywindow, mycolors(2), 8.0,
     .            style, 4, capstyle, 5, joinstyle, 5)
      IF ( mypen .EQ. 0.0 ) THEN
          WRITE(*,*) '68 FGDPEN returned a NULL pen'
          GOTO 900
      ENDIF
      ptsx(1) = 600
      ptsx(2) = 300
      ptsx(3) = 700
      ptsx(4) = 500
      ptsx(5) = 300
      ptsx(6) = 100
      ptsy(1) = 100
      ptsy(2) = 300
      ptsy(3) = 500
      ptsy(4) = 700
      ptsy(5) = 500
      ptsy(6) = 900
      CALL FGDDRAWMULTILINE(success, mywindow, ptsx, ptsy, 6, mypen)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '69 FGDDRAWMULTILINE failed'
          GOTO 900
      ENDIF
      CALL FGDPENDEL(success, mypen)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '70 FGDPENDEL failed'
          GOTO 900
      ENDIF
!     End of this view
      CALL FGDVIEWEND(success, mywindow)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '71 FGDVIEWEND failed'
          GOTO 900
      ENDIF
!     Window should be visible, but just be sure
      CALL FGDWINSETVIS(success, mywindow, 1)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '72 FGDWINSETVIS failed'
          GOTO 900
      ENDIF
      DO 100 K = 1, 17
          CALL FGDCOLORDEL(success, mycolors(k))
          IF ( success .EQ. 0 ) THEN
              WRITE(*,*) '73 - ', k, ' FGDCOLORDEL failed'
              GOTO 900
          ENDIF
 100  CONTINUE
      CALL FGDFONTDEL(success, myfont)
      IF ( success .EQ. 0 ) THEN
          WRITE(*,*) '74 FGDFONTDEL failed'
          GOTO 900
      ENDIF
      WRITE(*,*) "Success"
      RETURN

 900  errmsg = ' '
      CALL FGDERRMSG(errmsg, errmsglen)
      WRITE(*,*) errmsg(1:errmsglen)
      RETURN

      END

