C***********************************************************************
C***********************************************************************
C*		Copyright IBM Corporation 1989
C*
C*                      All Rights Reserved
C*
C* Permission to use, copy, modify, and distribute this software and its
C* documentation for any purpose and without fee is hereby granted,
C* provided that the above copyright notice appear in all copies and that
C* both that copyright notice and this permission notice appear in
C* supporting documentation, and that the name of IBM not be
C* used in advertising or publicity pertaining to distribution of the
C* software without specific, written prior permission.
C*
C* IBM DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
C* ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
C* IBM BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
C* ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
C* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
C* ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
C* SOFTWARE.
C*
C* $Id$
C*
C*****
C*****   Product:     graPHIGS GKS-CO Demonstration Program
C*****
C*****   Module Name: gksdemo.f
C*****
C*****   Module Type: FORTRAN
C*****
C*****   Descriptive Name: GKS Demonstration Program
C*****
C*****   Module Function:  Demonstration package written to
C*****                     show the programming techniques of
C*****                     writing graphics applications using
C*****                     the device independent graphics GKS
C*****                     standard, via the graPHIGS GKS-CO
C*****                     implementation of that standard.
C*****
C*****
C*****
C*****   Internal Flow:       GKSDEMO
C*****     ___________________| | | |____________________
C*****     |             _______| |_______              |
C*****     |             |               |              |
C*****     |             |               |              |
C*****  MAPDEM         PRIMIT          COLOR          INTER
C*****         ________||||||________                  ||||_____
C*****         |        ||||        |                 / ||     |
C*****     DEMOPM  _____||||_____ DEMOGD             /  ||   INTDEL
C*****             |     ||     |                   /   ||____
C*****         DEMOPL ___||___ DEMOCA              /    |    |
C*****         _|     |      |                    /     |  INTTRA
C*****        |    DEMOTX   DEMOFA             INTCRE   |___
C*****    DRWHLS                ________________||||       |
C***** .....................    |      __________|||     INTINS
C***** .Common Subroutines .    |      |       ___||____
C***** . Used by Many:     .    |      |       |       |
C***** ..................... CREPLN  CREPMK  CREFA  CRETXT
C***** .  CRSEGM FINSHF    .     \        \   /       /
C***** .  RCHOI  RMENU     .      \________\ /_______/
C***** .  SETCOL SETUPF    .              CHCOL
C***** .....................
C*****
C*****
C*****
C*****   Entry Point: GKSDEM
C*****
C*****   Subroutines:  CHCOL  - Allows User to choose a color
C*****                 COLOR  - Demonstrates the Color Facilities
C*****                 CREFA  - Create Fill Area in Segment
C*****                 CREPLN - Create Polyline in Segment
C*****                 CREPMK - Create Polymarker in Segment
C*****                 CRSEGM - Create Pickable Segment for RETURN
C*****                 CRETXT - Create Text in Segment
C*****                 DEMOCA - Demonstrates cell arrays
C*****                 DEMOFA - Demonstrates fill areas
C*****                 DEMOGD - Demonstrates GDPs Available
C*****                 DEMOPL - Demonstrates polylines
C*****                 DEMOPM - Demonstrates polymarkers
C*****                 DEMOTX - Demonstrates the text primitive
C*****                 DRWHLS - Draws the wheels for the car frame
C*****                 FINSHF - Finishes the screen frame borders
C*****                 INTCRE - Create Segment
C*****                 INTDEL - Delete a Segment
C*****                 INTER  - Main Interactive Demo Menu
C*****                 INTINS - Insert Segment(s) into New Segment
C*****                 INTTRA - Transform Segment Data
C*****                 MAPDEM - Draws high level picture of GKS-CO
C*****                 PRIMIT - Main output primitive menu
C*****                 RCHOI  - Waits on input from choice device
C*****                 RMENU  - Sets up choice prompts
C*****                 SETCOL - Sets up the color table
C*****                 SETUPF - Draws the screen frame borders
C*****
C*****   Include Files:  gkspar.inc
C*****
C*****   Environment Dependencies: IBM RT PC with lpfks, tablet,
C*****                             and dial devices attached.
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE GKSDEM
      INCLUDE 'gkspar.inc'

C***********************************************************************
C*****
C*****   Declare all Variables
C*****
C***********************************************************************

      INTEGER*4     ERRFIL,WKID,CONID,WTYPE,TNR
      INTEGER*4     ERRIND,DCUNIT,LX,LY,NOERR
      INTEGER*4     I,J,IA(32),MENLEN(5),ICLASS,SGNA
      INTEGER*4     YES,NO,LDRCDU,LASP(13)

      REAL*4        RX,RY,STARTX,STARTY,PX(10),PY(10)
      REAL*4        SELBLK(8),ARROW(14),TXTBLK(8),EMPTRP(1)
      REAL*4        TXTX,TXTY,XMIN,XMAX,YMIN,YMAX

      CHARACTER*1   CH
      CHARACTER*1   CHAR
      CHARACTER*1   DUMMYA(10)
      CHARACTER*20  MENTXT(5)
      CHARACTER*80  DRCDUM(4)
      DOUBLEPRECISION        DDUMMY
      EQUIVALENCE   (DRCDUM(1),DDUMMY)

C***********************************************************************
C*****
C*****   Define COMMON block variables
C*****
C***********************************************************************

      COMMON        /WINF/ WKID,WTYPE
      COMMON        /LIMITS/ XMIN,XMAX,YMIN,YMAX
      COMMON        /TEXT/ TXTX,TXTY

C***********************************************************************
C*****
C*****   Initialize Variables
C*****
C***********************************************************************

      DATA ERRFIL /0/
      DATA CONID /1/
      DATA TNR /1/
      DATA LASP /1,1,1,1,1,1,1,1,1,1,1,1,1/
      DATA NOERR /0/
      DATA NO /0/
      DATA YES /1/

C***********************************************************************
C*****
C*****  Set up the coordinates for the select, arrow and text
C*****  blocks.
C*****
C***********************************************************************

      DATA SELBLK /0.000,0.100,0.100,0.000,0.100,0.100,0.000,0.000/
      DATA ARROW  /0.000,0.140,0.140,0.185,0.140,0.140,0.000,
     *             0.016,0.016,0.000,0.024,0.048,0.032,0.032/
      DATA TXTBLK /0.000,0.535,0.535,0.000,0.100,0.100,0.000,0.000/

C***********************************************************************
C*****
C*****  Text for the menu options and the length of each text
C*****  string.
C*****
C***********************************************************************

      DATA MENTXT(1)  /'XGKS'/,
     *     MENTXT(2)  /'PRIMITIVES'/,
     *     MENTXT(3)  /'COLOR'/,
     *     MENTXT(4)  /'INTERACTION'/,
     *     MENTXT(5)  /'END'/
      DATA MENLEN /4,10,5,11,3/

      DATA EMPTRP /0.0/
      DATA CHAR   /' '/

      WKID = 1
      WTYPE = 4
C***********************************************************************
C*****
C*****   Open GKS, open the workstation and activate it.
C*****   Set the deferral mode to 'bnil' which will produce a
C*****   visual effect of each function appearing before the next
C*****   interaction locally. Set the implicit regeneration mode
C*****   to suppressed.
C*****
C***********************************************************************

      CALL GOPKS(ERRFIL,0)
      CALL GOPWK(WKID,CONID,WTYPE)
      CALL GACWK(WKID)
      CALL GSDS(WKID,GBNIL,GSUPPD)

C***********************************************************************
C*****
C*****   Set all the aspect source flags to individual.
C*****   Set the normalization transformation for transformation
C*****   number one.
C*****
C***********************************************************************

      CALL GSASF(LASP)
      CALL GSWN(TNR,0.0,1.0,0.0,1.0)
      CALL GSVP(TNR,0.0,1.0,0.0,1.0)

C***********************************************************************
C*****
C*****    Set the workstation transformation after inquiring
C*****    the display information.
C*****
C***********************************************************************

      CALL GQDSP(WTYPE,ERRIND,DCUNIT,RX,RY,LX,LY)
      XMIN = (RX - RY) / 2.0
      XMAX = XMIN + RY
      YMIN = 0.000
      YMAX = RY
      CALL GSWKWN(WKID,0.0,1.0,0.0,1.0)
      CALL GSWKVP(WKID,XMIN,XMAX,YMIN,YMAX)
      CALL GSELNT(TNR)

C***********************************************************************
C*****
C*****   Call Subroutine SETCOL to set up the color table for
C*****   this program.
C*****
C***********************************************************************

      CALL SETCOL

C***********************************************************************
C*****
C*****   Build the main menu from which all paths are chosen.
C*****   Main loop: keep drawing the main menu until the user
C*****   chooses to exit the routine (by choosing the segment
C*****   'end' - segment 5) via the LPFK or pick device.
C*****
C***********************************************************************

C********************************************************
C*****
C*****   Call subroutine SETUPF to draw the menu
C*****   border. Add the text to the menu and call
C*****   subroutine FINSHF to complete the menu
C*****   border.
C*****
C********************************************************

   10 CALL SETUPF(.TRUE.,.FALSE.)
      CALL GSCHH(0.065)
      CALL GSCHSP(0.0)
      CALL GTXS(TXTX,0.835,9,'XGKS DEMO')
      CALL FINSHF(.TRUE.)

      CALL GSFAIS(GSOLID)
      CALL GSFACI(GYELOW)
      CALL GSTXFP(GFONT2,GCHARP)
      CALL GSTXCI(GRED)
      CALL GSCHH(0.060)
      CALL GSCHXP(0.9)
      CALL GSCHSP(0.0)

C********************************************************
C*****
C*****   Set all LPFKs to off (1).
C*****
C********************************************************

      DO 20 I=1,32
        IA(I)=GPROFF
   20 CONTINUE

C********************************************************
C*****
C*****   For each of the 5 selectable segments,
C*****   calculate the 4 points that indicate its
C*****   boundary.
C*****
C********************************************************

      DO 60 I=0,4
        STARTX = 0.065
        STARTY = 0.590 - I * 0.130
        DO 30 J=1,4
          PX(J) = SELBLK(J) + STARTX
          PY(J) = SELBLK(J+4) + STARTY
   30   CONTINUE

C********************************************************
C*****
C*****  Display pickable choices on the main menu.
C*****
C********************************************************

        CALL GCRSG(I+1)
        CALL GFA(4,PX,PY)
        WRITE(CH,'(I1)')I+1
        CALL GSTXAL(GACENT,GAHALF)
        CALL GTXS(STARTX+0.05,STARTY+0.05,1,CH)
        CALL GSTXAL(GAHNOR,GAVNOR)

C********************************************************
C*****
C*****   Set the LPFK for this segment to on (2).
C*****   Close the segment and make it detectable.
C*****
C********************************************************

        IA(I+1) = GPRON
        CALL GCLSG
        CALL GSDTEC(I+1,1)

C********************************************************
C*****
C*****   Draw arrows on menu to indicate which LPFK
C*****   goes with which choice.
C*****
C********************************************************

        STARTX = 0.190
        STARTY = 0.606 - I * 0.130
        DO 40 J = 1,7
          PX(J) = ARROW(J) + STARTX
          PY(J) = ARROW(J+7) + STARTY
   40   CONTINUE
        CALL GPL(7,PX,PY)
        STARTX = 0.400
        STARTY = 0.590 - I * 0.130
        DO 50 J=1,4
          PX(J) = TXTBLK(J) + STARTX
          PY(J) = TXTBLK(J+4) + STARTY
   50   CONTINUE

C********************************************************
C*****
C*****   Indicate choices for each segment as follows:
C*****      1 -> GKS-CO
C*****      2 -> Primitives
C*****      3 -> Color
C*****      4 -> Interaction
C*****      5 -> End
C*****
C********************************************************

        CALL GFA(4,PX,PY)
        STARTX = STARTX + 0.015
        STARTY = STARTY + 0.020
        CALL GTXS(STARTX,STARTY,MENLEN(I+1),MENTXT(I+1))
   60 CONTINUE

C********************************************************
C*****
C*****   Initialize both the choice (lpfks) and the
C*****   pick (tablet) input types. Set both devices
C*****   in input mode and put out a message to tell
C*****   the user to select an item.
C*****
C********************************************************

      CALL GSTXFP(GFONT1,GSTRKP)
      CALL GSCHSP(0.0)
      CALL GPREC(0,IA,0,EMPTRP,0,DUMMYA,CHAR,4,ERRIND,LDRCDU,DRCDUM)
      CALL GINPK(WKID,1,1,1,1,1,XMIN,XMAX,YMIN,YMAX,
     *           LDRCDU,DRCDUM)
      CALL GPREC(32,IA,0,EMPTRP,0,DUMMYA,CHAR,4,ERRIND,LDRCDU,DRCDUM)
      CALL GINCH(WKID,1,1,1,2,XMIN,XMAX,YMIN,YMAX,
     *           LDRCDU,DRCDUM)
      CALL GSCHM(WKID,1,2,1)
      CALL GSPKM(WKID,1,2,1)
      CALL GMSGS(WKID,31,'    PICK SQUARES TO SELECT ITEM')

C********************************************************
C*****
C*****   Loop forever until input from either the
C*****   lpfks or tablet is received.
C*****
C********************************************************

   70 CALL GWAIT(600.0,WKID,ICLASS,IDEV)
      IF (ICLASS.EQ.4) THEN
        CALL GGTCH(STAT,SGNA)
      ELSE
        IF (ICLASS.EQ.5) THEN
          CALL GGTPK(STAT,SGNA,PCID)
        ELSE
          GOTO 70
        ENDIF
      ENDIF

C********************************************************
C*****
C*****   First, prepare for further input by enabling
C*****   the choice and pick devices and then flushing
C*****   their queues.
C*****
C********************************************************

      CALL GSCHM(WKID,1,0,1)
      CALL GSPKM(WKID,1,0,1)
      CALL GFLUSH(WKID,4,1)
      CALL GFLUSH(WKID,5,1)
      CALL GMSGS(WKID,1,' ')
      STARTX = 0.190
      STARTY = 0.736 - SGNA * 0.130
      DO 80 J = 1,7
        PX(J) = ARROW(J) + STARTX
        PY(J) = ARROW(J+7) + STARTY
   80 CONTINUE
      CALL GMSGS(WKID,1,' ')

C********************************************************
C*****
C*****   Check input choice and call subroutine to
C*****   handle the request. If 'END' was selected,
C*****   exit the input loop and leave the program
C*****   gracefully.
C*****
C********************************************************

      IF (SGNA.EQ.1) THEN
        CALL MAPDEM
      ELSE
        IF (SGNA.EQ.2) THEN
          CALL PRIMIT
        ELSE
          IF (SGNA.EQ.3) THEN
            CALL COLOR
          ELSE
            IF (SGNA.EQ.4) THEN
              CALL INTER
            ELSE
              GOTO 90
            ENDIF
          ENDIF
        ENDIF
      ENDIF

C********************************************************
C*****
C*****   If 'END' was not selected, wait for next
C*****   choice from the main menu.
C*****
C********************************************************

      GOTO 10

C***********************************************************************
C*****
C*****   Otherwise, exit the program gracefully.
C*****   Deactivate and close the workstation, close gks.
C*****
C***********************************************************************

   90 CALL GDAWK(WKID)
      CALL GCLWK(WKID)
      CALL GCLKS
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: CHCOL
C*****
C*****   Subroutine Function: Choose a color from the color table.
C*****
C*****   Calls Subroutines:   RCHOI
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE CHCOL(COLI)
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,I,J,CHNR,COLI,ERRIND
      INTEGER*4    NELCOL,LCHCOL(16),LDRCOL

      REAL*4       STARTX,STARTY,PX(4),PY(4),COLBOX(8)
      REAL*4       XMIN,XMAX,YMIN,YMAX

      CHARACTER*2  MENCOL(16)
      CHARACTER*80 DTRCOL(16)

      COMMON      /WINF/   WKID,WTYPE
      COMMON      /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA MENCOL /'1','2','3','4','5','6','7','8','9','10','11',
     1             '12','13','14','15','16'/
      DATA LCHCOL /1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2/
      DATA NELCOL /16/

      DATA COLBOX /0.000,0.030,0.030,0.000,0.000,0.000,0.012,0.012/

C***********************************************************************
C*****
C*****   Draw the color table on the upper right hand side of the
C*****   screen using colored solid fill areas. Each color repre-
C*****   sents the corresponding choice number.
C*****
C***********************************************************************

      CALL GCRSG(1000)
      CALL GSFAIS(GSOLID)
      STARTX = 0.965
      DO 110 I=1,16
        CALL GSFACI(I)
        STARTY = 1.012 - I * 0.028
        DO 100 J=1,4
          PX(J) = COLBOX(J) + STARTX
          PY(J) = COLBOX(J+4) + STARTY
  100   CONTINUE
        CALL GFA(4,PX,PY)
  110 CONTINUE
      CALL GSFACI(GGREEN)
      CALL GSFAIS(GSOLID)
      CALL GCLSG

C***********************************************************************
C*****
C*****   Initialize the choice device to prompt for colors 1-16.
C*****   Call subroutine to wait for valid choice output and then
C*****   assign the color picked to 'COLI' and return it to the
C*****   calling routine.
C*****
C***********************************************************************

      CALL GPREC(0,0,0,0.0,NELCOL,LCHCOL,MENCOL,NELCOL,
     *           ERRIND,LDRCOL,DTRCOL)
      CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRCOL,DTRCOL)
      CALL GMSGS(WKID,18,'SELECT COLOR INDEX')
      CALL RCHOI(WKID,16,CHNR)
      COLI = CHNR
      CALL GDSG(1000)
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: COLOR
C*****
C*****   Subroutine Function: This subroutine will demonstrate the
C*****                        color facilities supported by GKS-CO
C*****                        for the given workstation.
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE COLOR
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,NUMELA,LCHSTR(5),LENGTH,ERRIND
      INTEGER*4    CRID,CGID,CBID,CHNR,I,STAT,SGNA,PCID,J,K
      INTEGER*4    TABLE(4,4)

      REAL*4       CR,CG,CB,XMIN,XMAX,YMIN,YMAX,PX(10),PY(10)
      REAL*4       FRAME1(10),FRAME2(10),FRAME3(10),FRAME4(10)
      REAL*4       COLBOX(8),INTBOX(8),MIXBOX(8),ARROW(14)
      REAL*4       FFX(4),FFY(4),GREEN(16),BLUE(16)

      LOGICAL      NORED,NOGRN,NOBLU,INVPK,CONT

      CHARACTER*16 MENU(5)
      CHARACTER*2  INDEX(16)
      CHARACTER*1  COLORI(3)
      CHARACTER*80 DTREC(5)

      COMMON       /WINF/   WKID,WTYPE
      COMMON       /TEXT/   TXTX,TXTY
      COMMON       /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA MENU   /'DEFINE COL','SET TABLE','RESET TABL','SPECTRUM',
     1             'RETURN'/
      DATA LCHSTR /10,9,10,8,6/
      DATA NUMELA /5/
      DATA FRAME1 /0.000,0.000,1.000,1.000,0.000,
     *             0.000,1.000,1.000,0.000,0.000/

      DATA FRAME2 /0.015,0.015,0.985,0.985,0.015,
     *             0.745,0.985,0.985,0.745,0.745/
      DATA FRAME3 /0.015,0.015,0.985,0.985,0.015,
     *             0.430,0.730,0.730,0.430,0.430/
      DATA FRAME4 /0.015,0.015,0.985,0.985,0.015,
     *             0.015,0.415,0.415,0.015,0.015/
      DATA COLBOX /0.001,0.050,0.050,0.001,0.001,0.001,0.130,0.130/
      DATA INTBOX /0.001,0.100,0.100,0.001,0.001,0.001,0.060,0.060/
      DATA MIXBOX /0.860,0.960,0.960,0.860,0.060,0.060,0.290,0.290/
      DATA ARROW  /0.000,0.150,0.150,0.200,0.150,0.150,0.000,
     *             0.016,0.016,0.000,0.024,0.048,0.032,0.032/
      DATA COLORI /'R','G','B'/
      DATA INDEX  /' 1',' 2',' 3',' 4',' 5',' 6',' 7',' 8',
     *             ' 9','10','11','12','13','14','15','16'/
      DATA TABLE  /15,11,07,03,
     *             14,10,06,02,
     *             13,09,05,01,
     *             12,08,04,16/
      DATA GREEN  /0.33,0.66,1.00,0.00,0.33,0.66,1.00,0.00,
     *             0.33,0.66,1.00,0.00,0.33,0.66,1.00,0.00/
      DATA BLUE   /0.00,0.00,0.00,0.33,0.33,0.33,0.33,0.66,
     *             0.66,0.66,0.66,1.00,1.00,1.00,1.00,0.00/

C***********************************************************************
C*****
C*****   Initialize the choice device and use pack data record to
C*****   display the prompts for the various choices:
C*****
C*****     Choice 1: Define a Color
C*****            2: Place Color in Current Color Table
C*****            3: Reset the Color Table to the Original
C*****            4: Display the Color Spectrum
C*****            5: Return
C*****
C***********************************************************************

      CALL GPREC(0,0,0,0.0,NUMELA,LCHSTR,MENU,NUMELA,
     *           ERRIND,LENGTH,DTREC)
      CALL GCLRWK(WKID,1)
  100 CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LENGTH,DTREC)

C***********************************************************************
C*****
C*****   Draw the borders and text for the demo menu frame.
C*****
C***********************************************************************

      CALL GCRSG(1099)
      CALL GSPLCI(GYELOW)
      CALL GPL(5,FRAME1(1),FRAME1(6))
      CALL GPL(5,FRAME2(1),FRAME2(6))
      CALL GSTXFP(GFONT2,GSTRKP)
      CALL GSCHSP(0.0)
      CALL GSCHH(0.080)
      CALL GSTXCI(GYELOW)
      CALL GSTXAL(GACENT,GAVNOR)
      CALL GTXS(0.5,0.885,5,'COLOR')
      CALL GTXS(0.5,0.765,10,'FACILITIES')
      CALL GPL(5,FRAME3(1),FRAME3(6))

C***********************************************************************
C*****
C*****   Draw the current color table, making each fill area that
C*****   represents an index into the table a segment that can be
C*****   selected via the locator later in the program.
C*****
C***********************************************************************

      CALL GSCHH(0.080)
      CALL GTXS(0.5,0.630,11,'COLOR TABLE')
      CALL GSTXAL(GAHNOR,GAVNOR)
      CALL GSTXFP(GFONT1,GCHARP)
      CALL GSCHH(0.030)
      CALL GSFAIS(GSOLID)
      STARTY = 0.480
      CALL GCLSG
      DO 120 I=1,16
        CALL GSFACI(I)
        STARTX = I * 0.060 - 0.035
        DO  110 J=1,4
          PX(J) = COLBOX(J) + STARTX
          PY(J) = COLBOX(J+4) + STARTY
  110   CONTINUE
        CALL GCRSG(I)
        CALL GTXS(PX(1)+0.005,PY(1)-0.040,2,INDEX(I))

        CALL GFA(4,PX,PY)
        CALL GCLSG
        CALL GSDTEC(I,GDETEC)
  120 CONTINUE

C***********************************************************************
C*****
C*****   Draw the color selection area on the lower half of the
C*****   display. Place each color fill area in a segment so that
C*****   it too can be selected by the locator device later on in
C*****   the program.
C*****
C***********************************************************************

      CALL GCRSG(1098)
      CALL GPL(5,FRAME4(1),FRAME4(6))
      CALL GTXS(0.140,0.310,4,'  0%')
      CALL GTXS(0.265,0.310,4,' 33%')
      CALL GTXS(0.390,0.310,4,' 66%')
      CALL GTXS(0.515,0.310,4,'100%')
      CALL GSTXFP(GFONT2,GCHARP)
      CALL GSCHH(0.040)
      CALL GSLWSC(2.0)
      CALL GCLSG
      DO 160 I=1,3
        CALL GSTXCI(I)
        CALL GSFACI(I)
        CALL GSPLCI(I)
        STARTY = 0.315 - I * 0.085
	CALL GCRSG(1098 - I)
        CALL GTXS(0.060,STARTY+0.010,1,COLORI(I))
	CALL GCLSG
        DO 140 J=1,4
          STARTX = J * 0.125 + 0.010
          DO 130 K=1,4
            PX(K) = INTBOX(K) + STARTX
            PY(K) = INTBOX(K+4) + STARTY
  130     CONTINUE
          SGNA = I * 4 + J + 16
          CALL GCRSG(SGNA)
          CALL GFA(4,PX,PY)
          CALL GCLSG
          CALL GSDTEC(SGNA,GDETEC)
  140   CONTINUE

C***********************************************************************
C*****
C*****   Calculate and draw the three arrows that point to the
C*****   fill area which will contain the resulting color once
C*****   one has been chosen.
C*****
C***********************************************************************

        DO 150 J=1,7
          PX(J) = ARROW(J) + 0.635
          PY(J) = ARROW(J+7) + STARTY + 0.006
  150   CONTINUE
	CALL GCRSG(1080-I)
        CALL GPL(7,PX,PY)
	CALL GCLSG
  160 CONTINUE

C***********************************************************************
C*****
C*****   To demonstrate how the table works, show the color
C*****   selection for 0.66 of Red, Green and Blue in the
C*****   resulting color box and make each of their chooseable
C*****   segments invisible. This completes the drawing of the
C*****   initial demo frame.
C*****
C***********************************************************************

      CRID = 23
      CGID = 27
      CBID = 31
      CR = 0.66
      CG = 0.66
      CB = 0.66
      CALL GSVIS(CRID,GINVIS)
      CALL GSVIS(CGID,GINVIS)
      CALL GSVIS(CBID,GINVIS)
      CALL GSFACI(GGRAY)
      CALL GCRSG(1081)
      CALL GFA(4,MIXBOX(1),MIXBOX(5))
      CALL GCLSG

C***********************************************************************
C*****
C*****   Call subroutine RCHOI to wait for a valid choice from
C*****   the choice device.
C*****
C***********************************************************************

  170 CALL GMSGS(WKID,1,' ')
      CALL RCHOI(WKID,5,CHNR)

C***********************************************************************
C*****
C*****   If the choice was to define a color, display a message
C*****   for the user to pick three color intensities.
C*****
C***********************************************************************

      IF (CHNR.EQ.1) THEN
        CALL GMSG(WKID,'  PICK 3 COLOR INTENSITIES')
        CALL GSVIS(CRID,GVISI)
        CALL GSVIS(CGID,GVISI)
        CALL GSVIS(CBID,GVISI)
        NORED = .TRUE.
        NOGRN = .TRUE.
        NOBLU = .TRUE.
        INVPK = .FALSE.

C***********************************************************************
C*****
C*****   Wait for three valid choices from the pick device - one
C*****   must be an intensity for red, one for green and one for
C*****   blue. Save the intensities choosen in CR,CG,and CB.
C*****
C***********************************************************************

        DO 190 I=1,3
  180     CALL GRQPK(WKID,1,STAT,SGNA,PCID)
          IF (STAT.EQ.1) THEN
            IF ((SGNA.GE.21).AND.(SGNA.LE.24)) THEN
              IF (NORED) THEN
                CRID = SGNA
                CR = REAL(SGNA - 21) / 3.0
                NORED = .FALSE.
              ELSE
                INVPK = .TRUE.
              ENDIF
            ELSE
              IF ((SGNA.GE.25).AND.(SGNA.LE.28)) THEN
                IF (NOGRN) THEN
                  CGID = SGNA
                  CG = REAL(SGNA - 25) / 3.0
                  NOGRN = .FALSE.
                ELSE
                  INVPK = .TRUE.
                ENDIF
              ELSE
                IF ((SGNA.GE.29).AND.(SGNA.LE.32)) THEN
                  IF (NOBLU) THEN
                    CBID = SGNA
                    CB = REAL(SGNA - 29) / 3.0
                    NOBLU = .FALSE.
                  ELSE
                    INVPK = .TRUE.
                  ENDIF
                ELSE
                  INVPK = .TRUE.
                ENDIF
              ENDIF
            ENDIF
          ELSE
            INVPK = .TRUE.
          ENDIF

C***********************************************************************
C*****
C*****   If a valid intensity was not picked, wait for more input
C*****   from the pick device. Otherwise, set the fill area segment
C*****   for the intensity chosen to invisible.
C*****
C***********************************************************************

          IF (INVPK) THEN
            INVPK = .FALSE.
            GOTO 180
          ELSE
            CALL GSVIS(SGNA,GINVIS)
          ENDIF
  190   CONTINUE

C***********************************************************************
C*****
C*****   Set the color representation for index 14 and draw the
C*****   resulting color in both the resultant fill area (MIXBOX)
C*****   and in index 14 of the main color table displayed above.
C*****
C***********************************************************************

        CALL GSCR(WKID,14,CR,CG,CB)
      ELSE

C***********************************************************************
C*****
C*****   If the choice was to assign the color to an index, allow
C*****   the user to pick one of the color squares in the color
C*****   table shown above.
C*****
C***********************************************************************

        IF (CHNR.EQ.2) THEN
          CALL GMSGS(WKID,23,'  ASSIGN COLOR TO INDEX')
  200     CALL GRQPK(WKID,1,STAT,SGNA,PCID)
          IF (STAT.LE.1) THEN

C***********************************************************************
C*****
C*****   If a valid color index was picked, set the user defined
C*****   color to that index. Draw the fill area with the new
C*****   color.
C*****
C***********************************************************************

            IF (SGNA.GE.1 .AND. SGNA.LE.16) THEN
              CALL GSCR(WKID,SGNA,CR,CG,CB)
            ENDIF
          ELSE
            GOTO 200
          ENDIF
        ELSE

C***********************************************************************
C*****
C*****   If the choice was to reset the color table, reset all the
C*****   current variables and call subroutine SETCOL to define the
C*****   color table in its initial state.
C*****
C***********************************************************************

          IF (CHNR.EQ.3) THEN
            CALL GMSGS(WKID,19,'  COLOR TABLE RESET')
            CALL SETCOL(WKID)
            CALL GSVIS(CRID,GVISI)
            CALL GSVIS(CGID,GVISI)
            CALL GSVIS(CBID,GVISI)
            CRID = 23
            CGID = 27
            CBID = 31
            CR = 0.66
            CG = 0.66
            CB = 0.66
            CALL GSVIS(CRID,GINVIS)
            CALL GSVIS(CGID,GINVIS)
            CALL GSVIS(CBID,GINVIS)
          ELSE

C***********************************************************************
C*****
C*****   If the choice was to display the color spectrum, draw a
C*****   demo frame that depicts the various colors available with
C*****   various intensities.
C*****
C***********************************************************************

            IF (CHNR.EQ.4) THEN
              CALL SETUPF(.TRUE.,.TRUE.)
              CALL GSTXCI(GWHITE)
              CALL GTXS(TXTX,TXTY,5,'COLOR')
              CALL FINSHF(.TRUE.)
              DO 210 I=1,15
                CALL GSCR(WKID,I,0.00,GREEN(I),BLUE(I))
  210         CONTINUE
              CALL GSTXFP(GFONT1,GSTRKP)
              CALL GSTXCI(GWHITE)
              CALL GSCHH(0.040)
              CALL GTXS(0.200,0.650,9,'RED:   0%')
              CALL GTXS(0.440,0.580,5,'BLUE:')
              CALL GTXS(0.025,0.280,6,'GREEN:')
              CALL GSCHH(0.020)
              CALL GTXS(0.300,0.520,23,' 100%   66%   33%    0%')
              CALL GTXS(0.200,0.440,4,'100%')
              CALL GTXS(0.200,0.340,4,' 66%')
              CALL GTXS(0.200,0.240,4,' 33%')
              CALL GTXS(0.200,0.140,4,'  0%')
              CALL GCA(0.300,0.100,0.700,0.500,4,4,1,1,4,4,TABLE)
              FFX(1) = 0.3
              FFY(1) = 0.4
              FFX(2) = 0.4
              FFY(2) = 0.4
              FFX(3) = 0.4
              FFY(3) = 0.5
              FFX(4) = 0.3
              FFY(4) = 0.5
              DO 230 I=1,4
                DO 220 J=1,4
                  CALL GSFACI(TABLE(J,I))
                  CALL GFA(4,FFX,FFY)
                  FFX(1) = FFX(1) + 0.1
                  FFX(2) = FFX(2) + 0.1
                  FFX(3) = FFX(3) + 0.1
                  FFX(4) = FFX(4) + 0.1
  220           CONTINUE
                FFX(1) = 0.3
                FFY(1) = FFY(1) - 0.1
                FFX(2) = 0.4
                FFY(2) = FFY(2) - 0.1
                FFX(3) = 0.4
                FFY(3) = FFY(3) - 0.1
                FFX(4) = 0.3
                FFY(4) = FFY(4) - 0.1
  230         CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

              CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the color spectrum,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the color facilities main screen.
C*****
C***********************************************************************

              IF (CONT) THEN
                CALL GSCHH(0.040)
                CALL GSTXCI(GBLACK)
                CALL GTXS(0.200,0.650,9,'RED:   0%')
                CALL GSTXCI(GWHITE)
                CALL GTXS(0.200,0.650,9,'RED:  33%')
                DO 240 I=1,15
                  CALL GSCR(WKID,I,0.33,GREEN(I),BLUE(I))
  240           CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

                CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the color spectrum,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the color facilities main screen.
C*****
C***********************************************************************

                IF (CONT) THEN
                  CALL GSTXCI(GBLACK)
                  CALL GTXS(0.200,0.650,9,'RED:  33%')
                  CALL GSTXCI(GWHITE)
                  CALL GTXS(0.200,0.650,9,'RED:  66%')
                  DO 250 I=1,15
                    CALL GSCR(WKID,I,0.66,GREEN(I),BLUE(I))
  250             CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

                  CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the color spectrum,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the color facilities main screen.
C*****
C***********************************************************************

                  IF (CONT) THEN
                    CALL GSTXCI(GBLACK)
                    CALL GTXS(0.200,0.650,9,'RED:  66%')
                    CALL GSTXCI(GWHITE)
                    CALL GTXS(0.200,0.650,9,'RED: 100%')
                    DO 260 I=1,15
                      CALL GSCR(WKID,I,1.00,GREEN(I),BLUE(I))
  260               CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompt
C*****   for the lpfk (return). RMENU will call RCHOI to wait for
C*****   a valid choice selection and will pass that selection back
C*****   in CHNR. But in this case, the user could only select
C*****   return as a valid choice, so we will fall down to the
C*****   return statement and go back to the color facilities main
C*****   panel.
C*****
C***********************************************************************

                    CALL RMENU(.TRUE.,CONT)
                  ENDIF
                ENDIF
              ENDIF

C***********************************************************************
C*****
C*****   Reinitialize the color table and return to the color
C*****   facilities selection panel.
C*****
C***********************************************************************

              CALL GCLRWK(WKID,GALWAY)
              CALL SETCOL
              GOTO 100
            ELSE

C***********************************************************************
C*****
C*****   If the user selected return from the main panel, reset the
C*****   color table to its original state and return to the
C*****   selection frame of the main program.
C*****
C***********************************************************************

              IF (CHNR.EQ.5) THEN
                CALL SETCOL
                GOTO 270
              ENDIF
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      GOTO 170
  270 RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: CREFA
C*****
C*****   Subroutine Function: This subroutine allows the user to
C*****                        create fill areas in new segments.
C*****                        The user can choose to change the
C*****                        fill area color and interior style.
C*****                        The user can then draw the fill area
C*****                        shape using the stroke device.
C*****
C*****   Calls Subroutines:   RCHOI, CHCOL
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE CREFA
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,CHNR,ERRIND,STAT,TNR,I,J,IA(4),COLI
      INTEGER*4    NELFLA,LCHFLA(4),LDRFLA
      INTEGER*4    NELDUM,LCHDUM(1),LDRDUM
      INTEGER*4    NELFIS,LCHFIS(4),LDRFIS

      REAL*4       XMIN,XMAX,YMIN,YMAX,LCX(337),LCY(337)
      REAL*4       IPX(1),IPY(1),SKIVAL(4)

      CHARACTER*16 MENFLA(4)
      CHARACTER*80 DTRFLA(4)
      CHARACTER*8  MENFIS(4)
      CHARACTER*80 DTRFIS(4)
      CHARACTER*1  MENDUM

      LOGICAL      SGNAME(50)

      COMMON      /WINF/   WKID,WTYPE
C The AIX fortran compiler only allows common blocks to be
C dimensioned once. The following line was modifed for AIX port. c2071
      COMMON      /SEGM/   SGNAME
      COMMON      /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA MENDUM /' '/
      DATA LCHDUM /1/
      DATA NELDUM /4/

      DATA MENFLA /'FILL AREA','COLOR','INT. STYLE','RETURN'/
      DATA LCHFLA /9,5,10,6/
      DATA NELFLA /4/

      DATA MENFIS /'HOLLOW','SOLID','PATTERN','HATCH'/
      DATA LCHFIS /6,5,7,5/
      DATA NELFIS /4/

      DATA IA     /0,0,0,0/
      DATA SKIVAL /0.005,0.005,0.005,1.000/

C***********************************************************************
C*****
C*****   Use Pack Data Record to set up the prompts for the choice
C*****   device. The user can then choose to: draw fill area, set
C*****   the fill area color, set the fill area interior style,
C*****   or return to the main create panel.
C*****
C***********************************************************************

      CALL GPREC(0,0,0,0.0,NELFLA,LCHFLA,MENFLA,NELFLA,
     *           ERRIND,LDRFLA,DTRFLA)
  300 CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRFLA,DTRFLA)
      CALL GMSGS(WKID,32,'DRAW FILL AREA OR SET ATTRIBUTES')
      CALL RCHOI(WKID,4,CHNR)

C***********************************************************************
C*****
C*****   Find the first unused segment number (segment 1-6 are re-
C*****   served for use by GKSDEMO). Allow the user to draw the
C*****   fill area using the stroke device (user holds button one
C*****   down on the locator while drawing and releases the button
C*****   when fill area boundaries are complete).
C*****
C***********************************************************************

      IF (CHNR.EQ.1) THEN
        DO 310 I=7,50
          IF (.NOT.(SGNAME(I))) GOTO 320
  310   CONTINUE
        CALL GMSGS(WKID,32,'NO MORE THAN 50 SEGMENTS ALLOWED')
        GOTO 340
  320   J = 10
        IPX(1) = 0.0
        IPY(1) = 0.0
	IA(1) = 0
	IA(2) = 1
        CALL GPREC(2,IA,3,SKIVAL,0,LCHDUM,MENDUM,NELDUM,
     *             ERRIND,LDRDUM,DTRDUM)
        CALL GINSK(WKID,1,1,1,IPX,IPY,1,XMIN,XMAX,YMIN,YMAX,
     *             337,LDRDUM,DTRDUM)
        CALL GMSGS(WKID,20,'ENTER POINT OR BREAK')
  330   CALL GRQSK(WKID,1,337,STAT,TNR,J,LCX,LCY)
        CALL GMSGS(WKID,1,' ')

C***********************************************************************
C*****
C*****   Once valid stroke data is received, create a segment to
C*****   hold the fill area and then place the fill area in the
C*****   segment. If valid stroke data wasn't received, try ob-
C*****   taining valid input again.
C*****
C***********************************************************************

        IF (STAT.EQ.1) THEN
          CALL GCRSG(I)
          CALL GSDTEC(I,1)
          SGNAME(I) = .TRUE.
          CALL GFA(J,LCX,LCY)
          CALL GCLSG
        ELSE
          GOTO 330
        ENDIF
      ELSE

C***********************************************************************
C*****
C*****   Call subroutine CHCOL to display a panel and allow the
C*****   user to choose a color from the current color table (color
C*****   index passed back in variable COLI). Set the fill area
C*****   color using the color index returned.
C*****
C***********************************************************************

        IF (CHNR.EQ.2) THEN
          CALL CHCOL(COLI)
          CALL GSFACI(COLI)
        ELSE

C***********************************************************************
C*****
C*****   Use pack data record to set up the prompt for the choice
C*****   device. User can choose an interior style of either:
C*****   hollow, solid, pattern, or hatch. When a valid choice
C*****   has been made, set the fill area interior style based on
C*****   the choice number minus (0=hollow,1=solid,2=pattern,
C*****   3=hatch).
C*****
C***********************************************************************

          IF (CHNR.EQ.3) THEN
            CALL GPREC(0,0,0,0.0,NELFIS,LCHFIS,MENFIS,NELFIS,
     *                 ERRIND,LDRFIS,DTRFIS)
            CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRFIS,
     *                 DTRFIS)
            CALL GMSGS(WKID,21,'SELECT INTERIOR STYLE')
            CALL RCHOI(WKID,4,CHNR)
            CALL GSFAIS(CHNR-1)

C***********************************************************************
C*****
C*****   User chose lpfk 4 - RETURN. So exit back to the main
C*****   create segment choice menu.
C*****
C***********************************************************************

          ELSE
            GOTO 340
          ENDIF
        ENDIF
      ENDIF
      GOTO 300
  340 RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: CREPLN
C*****
C*****   Subroutine Function: This subroutine allows the user to
C*****                        create a polyline in a new segment.
C*****                        The user can choose to change the
C*****                        polyline color, linewidth and line-
C*****                        type. The user can then use the
C*****                        stroke device to draw the line.
C*****
C*****   Calls Subroutines:   RCHOI, CHCOL
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE CREPLN
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,CHNR,ERRIND,STAT,TNR,I,J,IA(4),COLI
      INTEGER*4    NELLIN,LCHLIN(5),LDRLIN
      INTEGER*4    NELDUM,LCHDUM(1),LDRDUM
      INTEGER*4    NELCLT,LCHCLT(4),LDRCLT

      REAL*4       XMIN,XMAX,YMIN,YMAX,SCFACT
      REAL*4       IPX(1),IPY(1),LCX(337),LCY(337),SKIVAL(4)

      CHARACTER*12 MENCLT(4)
      CHARACTER*80 DTRCLT(4)
      CHARACTER*12 MENLIN(5)
      CHARACTER*80 DTRLIN(4)
      CHARACTER*1  MENDUM
      CHARACTER*80 DTRDUM(4)

      LOGICAL      SGNAME(50)

      COMMON      /WINF/   WKID,WTYPE
C The AIX fortran compiler only allows common blocks to be
C dimensioned once. The following line was modifed for AIX port. c2071
      COMMON      /SEGM/   SGNAME
      COMMON      /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA MENLIN /'POLYLINE','COLOR','LINEWIDTH','LINETYPE','RETURN'/
      DATA LCHLIN /8,5,9,8,6/
      DATA NELLIN /5/

      DATA MENDUM /' '/
      DATA LCHDUM /1/
      DATA NELDUM /4/

      DATA MENCLT /'SOLID','DASHED','DOTTED','DASHDOTTED'/
      DATA LCHCLT /5,6,6,10/
      DATA NELCLT /4/

      DATA IA     /0,0,0,0/
      DATA SKIVAL /0.005,0.005,0.005,1.000/

C***********************************************************************
C*****
C*****   Use Pack Data Record to set up the prompts for the choice
C*****   device. The user can then choose to: draw a polyline, set
C*****   the polyline color, set the linewidth, set the linetype,
C*****   or return to the main create panel.
C*****
C***********************************************************************

      CALL GPREC(0,0,0,0.0,NELLIN,LCHLIN,MENLIN,NELLIN,
     *           ERRIND,LDRLIN,DTRLIN)
  400 CALL GMSGS(WKID,1,' ')
      CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRLIN,DTRLIN)
      CALL GMSGS(WKID,31,'DRAW POLYLINE OR SET ATTRIBUTES')
      CALL RCHOI(WKID,5,CHNR)

C***********************************************************************
C*****
C*****   Find the first unused segment number (segment 1-6 are re-
C*****   served for use by GKSDEMO). Allow the user to draw the
C*****   polyline using the stroke device (user holds button one
C*****   down on the locator while drawing and releases the button
C*****   when polyline is complete).
C*****
C***********************************************************************

      IF (CHNR.EQ.1) THEN
        DO 410 I=7,50
          IF (.NOT.(SGNAME(I))) GOTO 420
  410   CONTINUE
        CALL GMSGS(WKID,32,'NO MORE THAN 50 SEGMENTS ALLOWED')
        GOTO 440
C demo modification  TRG  J = 10 not J = 0
  420   J = 10
        IPX(1) = 0.5
        IPY(1) = 0.5
	IA(1) = 0
	IA(2) = 1
        CALL GPREC(2,IA,3,SKIVAL,0,LCHDUM,MENDUM,NELDUM,
     *             ERRIND,LDRDUM,DTRDUM)
        CALL GINSK(WKID,1,1,1,IPX,IPY,1,XMIN,XMAX,YMIN,YMAX,
     *             337,LDRDUM,DTRDUM)
        CALL GMSGS(WKID,20,'ENTER POINT OR BREAK')
  430   CALL GRQSK(WKID,1,337,STAT,TNR,J,LCX,LCY)
        CALL GMSGS(WKID,1,' ')

C***********************************************************************
C*****
C*****   Once valid stroke data is received, create a segment to
C*****   hold the polyline and then place the polyline in the
C*****   segment. If valid stroke data wasn't received, try ob-
C*****   taining valid input again.
C*****
C***********************************************************************

        IF (STAT.EQ.1) THEN
          CALL GCRSG(I)
          CALL GSDTEC(I,1)
          SGNAME(I) = .TRUE.
          CALL GPL(J,LCX,LCY)
          CALL GCLSG
        ELSE
          GOTO 430
        ENDIF
      ELSE

C***********************************************************************
C*****
C*****   Call subroutine CHCOL to display a panel and allow the
C*****   user to choose a color from the current color table (color
C*****   index passed back in variable COLI). Set the polyline
C*****   color using the color index returned.
C*****
C***********************************************************************

        IF (CHNR.EQ.2) THEN
          CALL CHCOL(COLI)
          CALL GSPLCI(COLI)
        ELSE

C***********************************************************************
C*****
C*****   Use the valuator device to select the linewidth scale
C*****   factor (user turns valuator to desired position and then
C*****   hits the enter key).
C*****
C***********************************************************************

          IF (CHNR.EQ.3) THEN
            CALL GMSGS(WKID,31,'EVALUATE LINEWIDTH SCALE FACTOR')
            CALL GRQVL(WKID,1,STAT,SCFACT)
            CALL GMSGS(WKID,1,' ')
            CALL GSLWSC(SCFACT)
          ELSE

C***********************************************************************
C*****
C*****   Use the choice device to select the linetype. Valid
C*****   choices are: solid, dashed, dotted, and dash-dotted.
C*****
C***********************************************************************

            IF (CHNR.EQ.4) THEN
              CALL GPREC(0,0,0,0.0,NELCLT,LCHCLT,MENCLT,NELCLT,
     *                   ERRIND,LDRCLT,DTRCLT)
              CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRCLT,
     *                   DTRCLT)
              CALL GMSGS(WKID,15,'SELECT LINETYPE')
              CALL RCHOI(WKID,4,CHNR)
              CALL GSLN(CHNR)

C***********************************************************************
C*****
C*****   User chose lpfk 5 - RETURN. So exit back to the main
C*****   create segment choice menu.
C*****
C***********************************************************************

            ELSE
              GOTO 440
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      GOTO 400
  440 RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: CREPMK
C*****
C*****   Subroutine Function: This subroutine allows the user to
C*****                        create a polymarker in a new seg-
C*****                        ment. The user can choose to change
C*****                        the polymarker color, markersize
C*****                        and markertype. The user can then
C*****                        use the locator device to set the
C*****                        x,y coordinates of the polymarker
C*****                        and the polymarker segment is then
C*****                        created.
C*****
C*****   Calls Subroutines:   RCHOI,CHCOL
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE CREPMK
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,CHNR,ERRIND,STAT,TNR,I,IA(1),COLI
      INTEGER*4    NELDUM,LCHDUM(1),LDRDUM
      INTEGER*4    NELMRK,LCHMRK(5),LDRMRK
      INTEGER*4    NELMKT,LCHMKT(5),LDRMKT

      REAL*4       XMIN,XMAX,YMIN,YMAX,SCFACT
      REAL*4       LCX(337),LCY(337)

      CHARACTER*1  MENDUM
      CHARACTER*80 DTRDUM(4)
      CHARACTER*12 MENMRK(5)
      CHARACTER*80 DTRMRK(5)
      CHARACTER*12 MENMKT(5)
      CHARACTER*80 DTRMKT(5)

      LOGICAL      SGNAME(50)

      COMMON      /WINF/   WKID,WTYPE
C The AIX fortran compiler only allows common blocks to be
C dimensioned once. The following line was modifed for AIX port. c2071
      COMMON      /SEGM/   SGNAME
      COMMON      /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA MENDUM /' '/
      DATA LCHDUM /1/
      DATA NELDUM /4/

      DATA MENMRK /'POLYMARKER','COLOR','MARKERSIZE','MARKERTYPE',
     1             'RETURN'/
      DATA LCHMRK /10,5,10,10,6/
      DATA NELMRK /5/

      DATA MENMKT /'DOT','PLUS SIGN','ASTERISK','CIRCLE','DIAG-CROSS'/
      DATA LCHMKT /3,9,8,6,10/
      DATA NELMKT /5/

      DATA IA     /0/

C***********************************************************************
C*****
C*****   Use Pack Data Record to set up the prompts for the choice
C*****   device. The user can then choose to: draw fill area, set
C*****   the fill area color, set the fill area interior style,
C*****   or return to the main create panel.
C*****
C***********************************************************************

      CALL GPREC(0,0,0,0.0,NELMRK,LCHMRK,MENMRK,NELMRK,
     *           ERRIND,LDRMRK,DTRMRK)
  500 CALL GMSGS(WKID,1,' ')
      CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRMRK,DTRMRK)
      CALL GMSGS(WKID,33,'DRAW POLYMARKER OR SET ATTRIBUTES')
      CALL RCHOI(WKID,5,CHNR)

C***********************************************************************
C*****
C*****   Find the first unused segment number (segment 1-6 are re-
C*****   served for use by GKSDEMO). Allow the user to draw the
C*****   polymarker by placing the locator echo at a certain point
C*****   and hitting a button on the locator. When user is done
C*****   inserting polymarker segments, it hits ALT-CANCEL (break
C*****   in RT lingo). This will send it back to the main
C*****   polymarker choice menu.
C*****
C***********************************************************************

      IF (CHNR.EQ.1) THEN
        LCX(1) = 0.1
        LCY(1) = 0.1
  510   DO 520 I=7,50
          IF (.NOT.(SGNAME(I))) GOTO 530
  520   CONTINUE
        CALL GMSGS(WKID,32,'NO MORE THAN 50 SEGMENTS ALLOWED')
        GOTO 540
  530   CALL GPREC(0,IA,0,0.0,0,LCHDUM,MENDUM,NELDUM,ERRIND,
     *             LDRDUM,DTRDUM)
        CALL GINLC(WKID,1,1,LCX(1),LCY(1),2,XMIN,XMAX,YMIN,YMAX,
     *             LDRDUM,DTRDUM)
        CALL GMSGS(WKID,20,'ENTER POINT OR BREAK')
        CALL GRQLC(WKID,1,STAT,TNR,LCX(1),LCY(1))
        CALL GMSGS(WKID,1,' ')
        IF (STAT.EQ.1) THEN
          CALL GCRSG(I)
          CALL GSDTEC(I,1)
          SGNAME(I) = .TRUE.
          CALL GPM(1,LCX(1),LCY(1))
          CALL GCLSG
          GOTO 510
        ENDIF
      ELSE

C***********************************************************************
C*****
C*****   Call subroutine CHCOL to display a panel and allow the
C*****   user to choose a color from the current color table (color
C*****   index passed back in variable COLI). Set the polymarker
C*****   color using the color index returned.
C*****
C***********************************************************************

        IF (CHNR.EQ.2) THEN
          CALL CHCOL(COLI)
          CALL GSPMCI(COLI)
        ELSE
          IF (CHNR.EQ.3) THEN
            CALL GMSGS(WKID,32,'EVALUATE MARKERSIZE SCALE FACTOR')
            CALL GRQVL(WKID,1,STAT,SCFACT)
            CALL GMSGS(WKID,1,' ')
            CALL GSMKSC(SCFACT)
          ELSE
            IF (CHNR.EQ.4) THEN
              CALL GPREC(0,0,0,0.0,NELMKT,LCHMKT,MENMKT,NELMKT,
     *                   ERRIND,LDRMKT,DTRMKT)
              CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRMKT,
     *                   DTRMKT)
              CALL GMSGS(WKID,17,'SELECT MARKERTYPE')
              CALL RCHOI(WKID,5,CHNR)
              CALL GSMK(CHNR)

C***********************************************************************
C*****
C*****   User chose lpfk 5 - RETURN. So exit back to the main
C*****   create segment choice menu.
C*****
C***********************************************************************

            ELSE
              GOTO 540
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      GOTO 500
  540 RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: CRETXT
C*****
C*****   Subroutine Function: This subroutine allows the user to
C*****                        create text strings in new segments.
C*****                        The user can choose to change the
C*****                        text color, size and font. The user
C*****                        can then input the text from the
C*****                        keyboard and select the text starting
C*****                        point via the locator device.
C*****
C*****   Calls Subroutines:   RCHOI, CHCOL
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE CRETXT
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,CHNR,ERRIND,STAT,TNR,I,IA(1),COLI,LSTR
      INTEGER*4    NELTXT,LCHTXT(5),LDRTXT
      INTEGER*4    NELTXF,LCHTXF(11),LDRTXF
      INTEGER*4    NELDUM,LCHDUM(1),LDRDUM

      REAL*4       XMIN,XMAX,YMIN,YMAX,SCFACT
      REAL*4       LCX(337),LCY(337)

      CHARACTER*12 MENTXT(5)
      CHARACTER*80 DTRTXT(5)
      CHARACTER*8  MENTXF(11)
      CHARACTER*80 DTRTXF(11)
      CHARACTER*1  MENDUM
      CHARACTER*80 DTRDUM(4)
      CHARACTER*80 STRBUF

      LOGICAL      SGNAME(50)

      COMMON      /WINF/   WKID,WTYPE
C The AIX fortran compiler only allows common blocks to be
C dimensioned once. The following line was modifed for AIX port. c2071
      COMMON      /SEGM/   SGNAME
      COMMON      /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA MENTXT /'TEXT','COLOR','TEXTSIZE','TEXTFONT','RETURN'/
      DATA LCHTXT /4,5,8,8,6/
      DATA NELTXT /5/

      DATA MENTXF /'F  1','F  2','F  3','F  4','F  5','F  6','F  7',
     1             'F  8','F  9','F 10','F 11'/
      DATA LCHTXF /4,4,4,4,4,4,4,4,4,4,4/
      DATA NELTXF /11/

      DATA MENDUM /' '/
      DATA LCHDUM /1/
      DATA NELDUM /4/

      DATA IA     /0/

C***********************************************************************
C*****
C*****   Use Pack Data Record to set up the prompts for the choice
C*****   device. The user can then choose to: input text and pick
C*****   its starting point, set the text color, set the text size,
C*****   select the text font, or return to the main create segment
C*****   panel.
C*****
C***********************************************************************

      CALL GPREC(0,0,0,0.0,NELTXT,LCHTXT,MENTXT,NELTXT,
     *           ERRIND,LDRTXT,DTRTXT)
  600 CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRTXT,DTRTXT)
      CALL GMSGS(WKID,27,'DRAW TEXT OR SET ATTRIBUTES')
      CALL RCHOI(WKID,5,CHNR)

C***********************************************************************
C*****
C*****   Find the first free segment number (segments 1-6 are
C*****   reserved for use by GKSDEMO). Request input from the
C*****   string device (keyboard) for the text to be displayed.
C*****   Find the starting point for the text by requesting input
C*****   from the locator device. Draw the new segment on the
C*****   screen and return to the text choice panel.
C*****
C***********************************************************************

      IF (CHNR.EQ.1) THEN
        DO 610 I=7,50
          IF (.NOT.(SGNAME(I))) GOTO 620
  610   CONTINUE
        CALL GMSGS(WKID,32,'NO MORE THAN 50 SEGMENTS ALLOWED')
        GOTO 640
  620   CALL GRQST(WKID,1,STAT,LSTR,STRBUF)
        CALL GMSGS(WKID,1,' ')
        CALL GPREC(0,IA,0,0.0,0,LCHDUM,MENDUM,NELDUM,
     *             ERRIND,LDRDUM,DTRDUM)
        CALL GINLC(WKID,1,1,0.1,0.1,2,XMIN,XMAX,YMIN,YMAX,
     *             LDRDUM,DTRDUM)
        CALL GMSGS(WKID,25,'ENTER TEXT STARTING POINT')
  630   CALL GRQLC(WKID,1,STAT,TNR,LCX(1),LCY(1))
        CALL GMSGS(WKID,1,' ')
        IF (STAT.EQ.1) THEN
          CALL GCRSG(I)
          CALL GSDTEC(I,1)
          SGNAME(I) = .TRUE.
          CALL GTXS(LCX(1),LCY(1),LSTR,STRBUF)
          CALL GCLSG
          GOTO 600
        ELSE
          GOTO 630
        ENDIF
      ELSE

C***********************************************************************
C*****
C*****   Call subroutine CHCOL to display a panel and allow the
C*****   user to choose a color from the current color table (color
C*****   index passed back in variable COLI). Set the text color
C*****   using the color index returned.
C*****
C***********************************************************************

        IF (CHNR.EQ.2) THEN
          CALL CHCOL(COLI)
          CALL GSTXCI(COLI)
        ELSE

C***********************************************************************
C*****
C*****   Obtain the character height expansion factor from the
C*****   valuator (user turns valuator to desired character height
C*****   and hits ENTER). Return to text choice panel.
C*****
C***********************************************************************

          IF (CHNR.EQ.3) THEN
            CALL GMSGS(WKID,25,'EVALUATE CHARACTER HEIGHT')
            CALL GRQVL(WKID,1,STAT,SCFACT)
            CALL GMSGS(WKID,1,' ')
            CALL GSCHH(0.1 * SCFACT)
          ELSE

C***********************************************************************
C*****
C*****   Use pack data record to set up the prompts for the choice
C*****   device. Call subroutine RCHOI to wait for a valid selec-
C*****   tion and then set the text font using the value returned.
C*****   Return to the text choice panel.
C*****
C***********************************************************************

            IF (CHNR.EQ.4) THEN
              CALL GPREC(0,0,0,0.0,NELTXF,LCHTXF,MENTXF,NELTXF,
     *                   ERRIND,LDRTXF,DTRTXF)
              CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRTXF,
     *                   DTRTXF)
              CALL GMSGS(WKID,16,'SELECT TEXT FONT')
              CALL RCHOI(WKID,11,CHNR)
              CALL GSTXFP(CHNR,GSTRKP)

C***********************************************************************
C*****
C*****   User chose lpfk 5 - RETURN. So exit back to the main
C*****   create segment choice menu.
C*****
C***********************************************************************

            ELSE
              GOTO 640
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      GOTO 600
  640 RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: CRSEGM
C*****
C*****   Subroutine Function: This subroutine will create a
C*****                        detectable segment (segment number
C*****                        6) and draw a red box with the word
C*****                        'RETURN' in it and then return to
C*****                        the caller. The caller is responsible
C*****                        for deleting segment 6 when it is
C*****                        finished using it.
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE CRSEGM
      INCLUDE 'gkspar.inc'

      REAL*4 PX(4),PY(4)
      DATA PX /0.01,0.15,0.15,0.01/
      DATA PY /0.94,0.94,0.99,0.99/

      CALL GCRSG(6)
      CALL GSDTEC(6,1)
      CALL GSFACI(GRED)
      CALL GSFAIS(GHOLLO)
      CALL GFA(4,PX,PY)
      CALL GSCHH(0.020)
      CALL GSTXCI(GRED)
      CALL GSTXAL(GACENT,GAHALF)
      CALL GTXS(0.07,0.965,6,'RETURN')
      CALL GSTXAL(GAHNOR,GAVNOR)
      CALL GSTXCI(GYELOW)
      CALL GSPLCI(GYELOW)
      CALL GCLSG
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: DEMOCA
C*****
C*****   Subroutine Function: Draw a castle tower and a tree to
C*****                        show how the cell array primitive
C*****                        and its attributes work.
C*****
C*****   Calls Subroutines:   SETUPF,FINSHF,RMENU
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE DEMOCA
      INCLUDE 'gkspar.inc'

      INTEGER*4   WKID,WTYPE,I
      INTEGER*4   PIXEL1(9,9),PIXEL2(9,9)
      REAL*4      X1,X2,Y1,Y2
      LOGICAL     CONT
      COMMON      /WINF/ WKID,WTYPE
      COMMON      /TEXT/   TXTX,TXTY
      DATA  PIXEL1 / 8, 8,13,13,13,13,13, 8, 8,
     *               8, 8, 8,13,13,13, 8, 8, 8,
     *               8, 8, 8,13,13,13, 8, 8, 8,
     *               8,12, 8,13,13,12, 8,12, 8,
     *              12,12,12,12,12,12,12,12,12,
     *              12,12,12,12,12,12,12,12,12,
     *              12,12,12,12,12,12,12,12,12,
     *               8,12,12,12,12,12,12,12, 8,
     *               8, 8,12,12,12,12,12, 8, 8/
      DATA  PIXEL2 / 3,11,11,11,11,11,11,11, 3,
     *               3, 3,11,11,11,11,11, 3, 3,
     *               3, 3,11,11, 3,11,11, 3, 3,
     *               3, 3,11,11, 3,11,11, 3, 3,
     *               3, 3,11,11, 3,11,11, 3, 3,
     *               3,11,11,11,11,11,11,11, 3,
     *              11,11,11,11,11,11,11,11,11,
     *              11,11,11,11,11,11,11,11,11,
     *              11, 3,11, 3,11, 3,11, 3,11/

C***********************************************************************
C*****
C*****   Call subroutines SETUPF and FINSHF to draw the border and
C*****   border and title of the demo frame. Then draw the first
C*****   panel which is a picture drawn using cell arrays.
C*****
C***********************************************************************

      CALL SETUPF(.TRUE.,.FALSE.)
      CALL GTXS(TXTX,TXTY,10,'CELL ARRAY')
      CALL FINSHF(.TRUE.)

C***********************************************************************
C*****
C*****   Use the values store in PIXEL2 array to display a castle
C*****   tower using the cell array primitive call.
C*****
C***********************************************************************

      CALL GCA(0.300,0.100,0.700,0.700,9,9,1,1,9,9,PIXEL2)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

      CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the cell array panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

      IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

        CALL SETUPF(.FALSE.,.FALSE.)
        CALL GTXS(TXTX,TXTY,16,'EXPAND RECTANGLE')
        CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Show how the cell array can be used easily to vary the
C*****   picture without changing the data.
C*****
C***********************************************************************

        Y1 = 0.150
        DO 700 I=1,4
          X1 = 0.500 - I * 0.050
          X2 = 0.500 + I * 0.050
          Y2 = 0.150 + I * 0.100
          CALL GCA(X1,Y1,X2,Y2,9,9,1,1,9,9,PIXEL1)
  700   CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

        CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the cell array panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

        IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

          CALL SETUPF(.FALSE.,.FALSE.)
          CALL GTXS(TXTX,TXTY,22,'DIFFERENT ASPECT RATIO')
          CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Draw three copies of the tree seen in the previous panel,
C*****   showing various aspect ratios.
C*****
C***********************************************************************

          CALL GCA(0.100,0.615,0.650,0.815,9,9,1,1,9,9,PIXEL1)
          CALL GCA(0.100,0.100,0.650,0.565,9,9,1,1,9,9,PIXEL1)
          CALL GCA(0.700,0.100,0.900,0.815,9,9,1,1,9,9,PIXEL1)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompt
C*****   for the lpfk (return). RMENU will call RCHOI to wait for
C*****   a valid choice selection and will pass that selection back
C*****   in CHNR. But in this case, the user could only select
C*****   return as a valid choice, so we will fall down to the
C*****   return statement and go back to the output primitves.
C*****
C***********************************************************************

          CALL RMENU(.TRUE.,CONT)
        ENDIF
      ENDIF

      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: DEMOFA
C*****
C*****   Subroutine Function: This subroutine demonstrates the
C*****                        fill area primitive and its
C*****                        attributes.
C*****
C*****   Calls Subroutines:   SETUPF,FINSHF,RMENU
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE DEMOFA
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,I,J,K,START,II
      REAL*4       STARTX,STARTY,ALPHA,PI,TM(2,3)
      REAL*4       PX(20),PY(20),RX(20),RY(20),BLOCK(8)
      REAL*4       EXOR3(6),RPLC3(6),EXOR4(8),RPLC4(8)
      REAL*4       EXOR5(10),RPLC5(10),XMIN,XMAX,YMIN,YMAX
      LOGICAL      CONT
      CHARACTER*10 TEXT(4)
      COMMON       /WINF/   WKID,WTYPE
      COMMON       /TEXT/   TXTX,TXTY
      COMMON       /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA  PI      /3.141593/

      DATA  TEXT(1) /'HOLLOW    '/
      DATA  TEXT(2) /'SOLID     '/
      DATA  TEXT(3) /'PATTERN   '/
      DATA  TEXT(4) /'HATCH     '/

      DATA  BLOCK /0.00,0.00,0.50,0.50,0.00,0.12,0.12,0.00/
      DATA  EXOR3 /0.10,0.40,0.40,0.15,0.15,0.75/
      DATA  RPLC3 /0.60,0.90,0.90,0.15,0.15,0.75/
      DATA  EXOR4 /0.05,0.45,0.45,0.05,0.10,0.10,0.40,0.40/
      DATA  RPLC4 /0.55,0.95,0.95,0.55,0.10,0.10,0.40,0.40/
      DATA  EXOR5 /0.15,0.35,0.45,0.25,0.05,0.25,0.25,0.45,0.55,0.45/
      DATA  RPLC5 /0.65,0.85,0.95,0.75,0.55,0.25,0.25,0.45,0.55,0.45/

C***********************************************************************
C*****
C*****   Call subroutines SETUPF and FINSHF to draw the border and
C*****   border and title of the demo frame. Then draw the first
C*****   panel which is a picture drawn using cell arrays.
C*****
C***********************************************************************

      CALL SETUPF(.TRUE.,.FALSE.)
      CALL GTXS(TXTX,TXTY,9,'FILL AREA')
      CALL FINSHF(.TRUE.)

C***********************************************************************
C*****
C*****   Draw a hot pink, fancy nine pointed star using the fill
C*****   area primitive with a solid interior style.
C*****
C***********************************************************************

      CALL GSFAIS(GSOLID)
      CALL GSFACI(GLMGNT)
      ALPHA = 0.0
      DO 800 I=1,9
        RX(I) = 0.500 + 0.350 * SIN(ALPHA)
        RY(I) = 0.375 + 0.350 * COS(ALPHA)
        ALPHA = ALPHA + 2.0 * PI / 9.0
  800 CONTINUE
      START = 0
      DO 810 I=1,9
        PX(I) = RX(START+1)
        PY(I) = RY(START+1)
        START = MOD(START+4,9)
  810 CONTINUE
      CALL GFA(9,PX,PY)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

      CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the cell array panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

      IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

        CALL SETUPF(.FALSE.,.FALSE.)
        CALL GTXS(TXTX,TXTY,15,'INTERIOR STYLES')
        CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Draw four rectangles on the right side of the frame to
C*****   show each of the interior styles for fill areas. Label
C*****   each style appropriately on the left hand side.
C*****
C***********************************************************************

        STARTX = 0.450
        CALL GSCHH(0.050)
        CALL GSTXAL(GAHNOR,GAVNOR)
        CALL GSTXFP(GFONT1,GSTRKP)
        DO 830 I=1,4
          STARTY = 0.950 - I * 0.220
          DO 820 J=1,4
            PX(J) = BLOCK(J) + STARTX
            PY(J) = BLOCK(J+4) + STARTY
  820     CONTINUE
          STARTY = STARTY + 0.050
          CALL GTXS(0.100,STARTY,10,TEXT(I))
          IF (I.EQ.4) THEN
            CALL GSFASI(-(4))
          ENDIF
          CALL GSFAIS(I-1)
          CALL GSFACI(I)
          CALL GFA(4,PX,PY)
  830   CONTINUE
        CALL GSFASI(1)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

        CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the cell array panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

        IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

          CALL SETUPF(.FALSE.,.FALSE.)
          CALL GTXS(TXTX,TXTY,15,'TRANSFORMATIONS')
          CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Show a picture of fill area transformations. Draw a
C*****   triangle and rotate it. Draw a hexagon and zoom it.
C*****
C***********************************************************************

          CALL GSCHH(0.050)
          CALL GSTXFP(GFONT1,GSTRKP)
          CALL GSFACI(GLGREN)
          CALL GCRSG(1)
          CALL GSFAIS(GPATTR)
          CALL GFA(3,EXOR3(1),EXOR3(4))
          CALL GCLSG
          CALL GCRSG(3)
          CALL GSFAIS(GSOLID)
          CALL GFA(3,RPLC3(1),RPLC3(4))
          CALL GSFACI(GLGREN)
          CALL GFA(5,EXOR5(1),EXOR5(6))
          CALL GCLSG
          CALL GSFAIS(GPATTR)
          CALL GCRSG(2)
          CALL GFA(5,RPLC5(1),RPLC5(6))
          CALL GCLSG
          CALL GSVP(1,0.015,0.985,0.015,0.985)
	  CALL GCRSG(5)
          CALL GTXS(0.550,0.825,8,'ROTATION')
          CALL GTXS(0.050,0.825,7,'ZOOMING')
	  CALL GCLSG
          CALL RMENU(.FALSE.,CONT)
          CALL GEVTM(0.250,0.450,0.000,0.000,PI/4.0,1.0,1.0,0,TM)
          CALL GSSGT(1,TM)
          CALL GEVTM(0.750,0.400,0.000,0.000,0.0,0.5,0.5,0,TM)
          CALL GSSGT(2,TM)
          CALL GSVP(1,0.000,1.000,0.000,1.000)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

          CALL RMENU(.FALSE.,CONT)
	  CALL GDSG(1259)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the text panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

          IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

            CALL SETUPF(.FALSE.,.FALSE.)
            CALL GTXS(TXTX,TXTY,12,'HATCH STYLES')
            CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Draw a screen full of all the different hatch styles that
C*****   are supported by GKS-CO.
C*****
C***********************************************************************

            CALL GSFAIS(GHATCH)
            CALL GSFACI (2)
            II = 0
            DO 850 K=1,5
              DO 840 J=1,4
                II = II + 1
                CALL GSFASI(-(II))
                PX(1) = 0.09 + (J-1) * 0.21
                PY(1) = 0.05 + (K-1) * 0.17
                PX(2) = PX(1) + 0.19
                PY(2) = PY(1)
                PX(3) = PX(2)
                PY(3) = PY(1) + 0.15
                PX(4) = PX(1)
                PY(4) = PY(3)
                CALL GFA(4,PX,PY)
  840         CONTINUE
  850       CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompt
C*****   for the lpfk (return). RMENU will call RCHOI to wait for
C*****   a valid choice selection and will pass that selection back
C*****   in CHNR. But in this case, the user could only select
C*****   return as a valid choice, so we will fall down to the
C*****   return statement and go back to the output primitves.
C*****
C***********************************************************************

            CALL RMENU(.TRUE.,CONT)
          ENDIF
        ENDIF
      ENDIF

      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: DEMOGD
C*****
C*****   Subroutine Function: As there are no Generalized Drawing
C*****                        primitives yet available in GKS-CO,
C*****                        this routine simply fills the screen
C*****                        with 'not available' and waits for
C*****                        the user to chose to go on.
C*****
C*****   Calls Subroutines:   SETUPF,FINSHF,RMENU
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE DEMOGD

      REAL*4       TXTX,TXTY
      LOGICAL      CONT
      COMMON       /TEXT/ TXTX,TXTY

      CALL SETUPF(.TRUE.,.FALSE.)
      CALL GTXS(TXTX,TXTY,5,'G D P')
      CALL FINSHF(.TRUE.)
      CALL GTXS(0.10,0.350,13,'NOT AVAILABLE')
      CALL RMENU(.TRUE.,CONT)
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: DEMOPL
C*****
C*****   Subroutine Function: Draw a car to demonstrate the
C*****                        polyline output primitive. Draw the
C*****                        different linetypes. Show the various
C*****                        line attributes on a demo frame. And
C*****                        draw two different pictures of the
C*****                        sunflower star by using the same
C*****                        x,y data but varying the viewport.
C*****
C*****   Calls Subroutines:   SETUPF,FINSHF,DRWHLS,RMENU
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE DEMOPL
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,N,I,ERRIND,NLT,LT,NLW,NPPLI,K

      REAL*4       NOMLW,RLWMIN,RLWMAX
      REAL*4       OX(28),OY(28),KFX(26),KFY(26),UX(6),UY(6)
      REAL*4       DTX(11),DTY(11),MITX,MITY,R,ALPHA,RX(21),RY(21)
      REAL*4       T1X(2),T1Y(2),T2X(2),T3X(2),T3Y(2),T4X(4),T4Y(4)
      REAL*4       GR1X(2),GR2X(2),GR1Y(2),VDX(3),VDY(3)
      REAL*4       K1X(3),K1Y(3),K2X(3),K2Y(3),K3X(2),K3Y(2)
      REAL*4       G1X(2),G2X(2),G3X(2),G4X(2),G5X(2),G6X(2)
      REAL*4       G7X(2),G8X(2),G1Y(2),G2Y(2),G3Y(2),G4Y(2)
      REAL*4       G5Y(2),G6Y(2),G7Y(2),G8Y(2),XMIN,XMAX,YMIN,YMAX
      REAL*4       KOX(11),KOY(11),FIX(9),FIY(9)
      REAL*4       PLX(2),PLY(2),RLX(20),RLY(20)

      LOGICAL      CONT

      COMMON       /WINF/   WKID,WTYPE
      COMMON       /TEXT/   TXTX,TXTY
      COMMON       /LIMITS/ XMIN,XMAX,YMIN,YMAX

C  CAR DATA

      DATA    OX   /0.264,0.272,0.770,0.776,0.774,0.608,0.594,0.588,
     *              0.564,0.330,0.320,0.310,0.300,0.290,0.280,0.268,
     *              0.264,0.258,0.250,0.240,0.234,0.228,0.224,0.222,
     *              0.224,0.280,0.288,0.290/
      DATA    OY   /0.396,0.398,0.400,0.406,0.416,0.418,0.464,0.466,
     *              0.468,0.468,0.466,0.462,0.456,0.445,0.430,0.400,
     *              0.396,0.394,0.390,0.380,0.370,0.348,0.320,0.296,
     *              0.286,0.284,0.286,0.304/
      DATA    KFX  /0.222,0.230,0.286,0.300,0.310,0.321,0.340,0.358,
     *              0.370,0.378,0.388,0.394,0.398,0.400,0.574,0.584,
     *              0.726,0.736,0.746,0.760,0.770,0.780,0.792,0.802,
     *              0.808,0.812/
      DATA    KFY  /0.296,0.300,0.350,0.360,0.366,0.370,0.372,0.370,
     *              0.364,0.358,0.346,0.332,0.316,0.294,0.294,0.297,
     *              0.371,0.373,0.374,0.372,0.370,0.366,0.358,0.346,
     *              0.332,0.310/
      DATA    UX   /0.382,0.384,0.390,0.690,0.698,0.702/
      DATA    UY   /0.302,0.286,0.282,0.282,0.290,0.302/
      DATA    DTX  /0.584,0.584,0.580,0.576,0.572,0.564,0.550,0.320,
     *              0.308,0.295,0.280/
      DATA    DTY  /0.297,0.399,0.424,0.440,0.448,0.452,0.454,0.454,
     *              0.452,0.448,0.430/
      DATA    T1X  /0.480,0.484/
      DATA    T1Y  /0.454,0.294/
      DATA    T2X  /0.476,0.480/
      DATA    T3X  /0.378,0.378/
      DATA    T3Y  /0.454,0.358/
      DATA    T4X  /0.584,0.590,0.598,0.608/
      DATA    T4Y  /0.399,0.406,0.414,0.418/
      DATA    GR1X /0.462,0.472/
      DATA    GR1Y /0.386,0.386/
      DATA    GR2X /0.488,0.498/
      DATA    VDX  /0.572,0.576,0.594/
      DATA    VDY  /0.448,0.454,0.464/
      DATA    K1X  /0.764,0.760,0.760/
      DATA    K1Y  /0.416,0.400,0.372/
      DATA    K2X  /0.774,0.770,0.770/
      DATA    K2Y  /0.416,0.400,0.369/
      DATA    K3X  /0.776,0.776/
      DATA    K3Y  /0.406,0.368/
      DATA    KOX  /0.793,0.795,0.812,0.816,0.824,0.822,0.816,0.806,
     *              0.796,0.786,0.776/
      DATA    KOY  /0.314,0.306,0.310,0.312,0.324,0.336,0.350,0.368,
     *              0.376,0.380,0.382/
      DATA    G1X  /0.670,0.672/
      DATA    G1Y  /0.390,0.378/
      DATA    G2X  /0.680,0.686/
      DATA    G2Y  /0.390,0.352/
      DATA    G3X  /0.690,0.696/
      DATA    G3Y  /0.390,0.357/
      DATA    G4X  /0.700,0.706/
      DATA    G4Y  /0.390,0.363/
      DATA    G5X  /0.710,0.715/
      DATA    G5Y  /0.390,0.367/
      DATA    G6X  /0.720,0.725/
      DATA    G6Y  /0.390,0.371/
      DATA    G7X  /0.730,0.735/
      DATA    G7Y  /0.390,0.373/
      DATA    G8X  /0.740,0.745/
      DATA    G8Y  /0.390,0.374/
      DATA  FIX /0.766,0.766,0.768,0.760,0.776,0.768,0.772,0.774,0.774/
      DATA  FIY /0.416,0.420,0.420,0.434,0.428,0.428,0.420,0.420,0.416/

C***********************************************************************
C*****
C*****   Inquire information about the polyline facilities provided
C*****   by GKS-CO. Call subroutines SETUPF and FINSHF to draw the
C*****   border and title of the demo frame. Then draw the car
C*****   using all the data provided. The car will be red, the
C*****   hood ornament yellow, and the wheels white.
C*****
C***********************************************************************


      CALL GQPLF(WTYPE,1,ERRIND,NLT,LT,NLW,NOMLW,RLWMIN,RLWMAX,NPPLI)

      CALL SETUPF(.TRUE.,.FALSE.)
      CALL GTXS(TXTX,TXTY,8,'POLYLINE')
      CALL FINSHF(.TRUE.)

      CALL GSWN(1,0.18,0.88,0.10,0.80)
      CALL GSPLCI(GRED)
      CALL GPL(28,OX,OY)
      CALL GPL(26,KFX,KFY)
      CALL GPL(6,UX,UY)
      CALL GPL(11,DTX,DTY)
      CALL GPL(11,KOX,KOY)
      CALL GPL(2,T1X,T1Y)
      CALL GPL(2,T2X,T1Y)
      CALL GPL(2,T3X,T3Y)
      CALL GPL(4,T4X,T4Y)
      CALL GPL(2,GR1X,GR1Y)
      CALL GPL(2,GR2X,GR1Y)
      CALL GPL(3,VDX,VDY)
      CALL GPL(3,K1X,K1Y)
      CALL GPL(3,K2X,K1Y)
      CALL GPL(2,K3X,K3Y)
      CALL GPL(2,G1X,G1Y)
      CALL GPL(2,G2X,G2Y)
      CALL GPL(2,G3X,G3Y)
      CALL GPL(2,G4X,G4Y)
      CALL GPL(2,G5X,G5Y)
      CALL GPL(2,G6X,G6Y)
      CALL GPL(2,G7X,G7Y)
      CALL GPL(2,G8X,G8Y)
      CALL GSPLCI(GYELOW)
      CALL GPL(9,FIX,FIY)

C***********************************************************************
C*****
C*****   Change the color to grayish white and call subroutine
C*****   DRWHLS to draw each of the car wheels - inside and outside
C*****   borders.
C*****
C***********************************************************************

      CALL GSPLCI(GGRAY)
      MITX = 0.336
      MITY = 0.302
      N = 14
      R = 0.020
      CALL DRWHLS(MITX,MITY,N,R)
      N = 20
      R = 0.044
      CALL DRWHLS(MITX,MITY,N,R)
      MITX = 0.748
      N = 14
      R = 0.020
      CALL DRWHLS(MITX,MITY,N,R)
      N = 20
      R = 0.044
      CALL DRWHLS(MITX,MITY,N,R)
      MITX = 0.640
      MITY = 0.350
      N = 14
      R = 0.020
      CALL DRWHLS(MITX,MITY,N,R)
      R = 0.044
      ALPHA = 3.80
      DO 900 I=1,14
        RX(I) = MITX + R * SIN(ALPHA)
        RY(I) = MITY + R * COS(ALPHA)
        ALPHA = ALPHA + 0.310
  900 CONTINUE
      CALL GPL(14,RX,RY)
      CALL GSWN(1,0.0,1.0,0.0,1.0)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

      CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the polyline panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

      IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

        CALL SETUPF(.FALSE.,.FALSE.)
        CALL GTXS(TXTX,TXTY,9,'LINETYPES')
        CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Draw a horizontal line in the linetype specified on the
C*****   left half of the demo frame for each linetype supported
C*****   by GKS-CO.
C*****
C***********************************************************************

        RLX(1) = 0.60
        PLX(1) = 0.05
        PLX(2) = 0.45
        CALL GSPLCI(GWHITE)
        DO 920 I=1,NLT
          CALL GSLN(I)
          PLY(1) = (0.90 - (0.98 * REAL(I) / REAL(NLT+1)))
          PLY(2) = PLY(1)
          CALL GPL(2,PLX,PLY)
          ALPHA = 0.0
          RLY(1) = PLY(1)

C***********************************************************************
C*****
C*****   Draw a fancy set of diagonal lines in each linetype.
C*****
C***********************************************************************

          DO 910 J=1,5
            RLX(2) = RLX(1) + 0.3 * SIN(ALPHA)
            RLY(2) = RLY(1) + (0.8 / REAL(NLT+1)) * COS(ALPHA)
            ALPHA = ALPHA + 3.141593 / 8.0
            CALL GPL(2,RLX,RLY)
  910     CONTINUE
  920   CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

        CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the polyline panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

        IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

          CALL SETUPF(.FALSE.,.FALSE.)
          CALL GTXS(TXTX,TXTY,10,'ATTRIBUTES')
          CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Draw on the left half of the demo frame, a line of each
C*****   linetype and linewidth supported.
C*****
C***********************************************************************

          RLX(1) = 0.60
          PLX(1) = 0.05
          RLX(1) = 0.45
          CALL GSPLCI(GWHITE)
          DO 930 I = 1,NPPLI
            CALL GSLN(I)
            CALL GSLWSC(REAL(I))
            PLY(1) = (0.92 - (0.92 * REAL(I) / REAL(NPPLI+1)))
            PLY(2) = PLY(1)
            CALL GPL(2,PLX,PLY)
  930     CONTINUE

C***********************************************************************
C*****
C*****   Draw on the right half of the demo frame, a star like
C*****   figure containing a line of each color supported by the
C*****   currently defined color table.
C*****
C***********************************************************************

          RLX(17) = 0.75
          RLY(17) = 0.50
          ALPHA = 0.0
          DO 940 I=1,16
            RLX(I) = RLX(17) + 0.2 * SIN(ALPHA)
            RLY(I) = RLY(17) + 0.2 * COS(ALPHA)
            ALPHA = ALPHA + 3.141593 / 8.0
  940     CONTINUE
          PLX(1) = RLX(17)
          PLY(1) = RLY(17)
          DO 950 I=1,16
            PLX(2) = RLX(I)
            PLY(2) = RLY(I)
            CALL GSLN(GLSOLI)
            CALL GSLWSC(1.0)
            IF (I.EQ.16) THEN
              CALL GSPLCI(GWHITE)
            ELSE
              CALL GSPLCI(I)
            ENDIF
            CALL GPL(2,PLX,PLY)
  950     CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

          CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the polyline panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

          IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

            CALL SETUPF(.FALSE.,.FALSE.)
            CALL GTXS(TXTX,TXTY,4,'STAR')
            CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Compute the coordinates of the sunflower star.
C*****
C***********************************************************************

            N = 13
            MITX = 0.50
            MITY = 0.45
            ALPHA = 0.0
            DO 960 I=1,N
              RX(I) = MITX + 0.4 * SIN(ALPHA)
              RY(I) = MITY + 0.4 * COS(ALPHA)
              ALPHA = ALPHA + 2.0 * 3.141593 / REAL(N)
  960       CONTINUE
            CALL GSPLCI(GLBLUE)
            DO 980 I=1,(N-1)/2
              DO 970 J=0,N
                K = MOD(I*J,N)
                RLX(J+1) = RX(K+1)
                RLY(J+1) = RY(K+1)
  970         CONTINUE

C***********************************************************************
C*****
C*****   Draw two stars on the demo frame, using the same x,y data
C*****   and varying the viewport.
C*****
C***********************************************************************

            CALL GSVP(1,0.00,0.60,0.35,0.95)
            CALL GPL(N+1,RLX,RLY)
            CALL GSVP(1,0.40,1.00,0.03,0.63)
            CALL GPL(N+1,RLX,RLY)
  980       CONTINUE

            CALL GSVP(1,0.0,1.0,0.0,1.0)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate one prompt
C*****   for the lpfk (return). RMENU will call RCHOI to wait for
C*****   a valid choice selection and will pass that selection back
C*****   in CHNR. But in this case, the user could only select
C*****   return as a valid choice, so we will fall down to the
C*****   return statement and go back to the output primitves.
C*****
C***********************************************************************

            CALL RMENU(.TRUE.,CONT)
          ENDIF
        ENDIF
      ENDIF

      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: DEMOPM
C*****
C*****   Subroutine Function: This subroutine demonstrates the
C*****                        polymarker output primitives and
C*****                        its attributes: color and scale
C*****                        factor.
C*****
C*****   Calls Subroutines:   SETUPF,FINSHF,RMENU
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE DEMOPM
      INCLUDE 'gkspar.inc'

      INTEGER*4   I
      REAL*4      PMX(1),PMY(1),TXTX,TXTY
      LOGICAL     CONT
      COMMON      /TEXT/   TXTX,TXTY

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   of the demo menu. Then use standard polymarker calls
C*****   to draw 3 rows of the various polymarker types in
C*****   different colors and sizes.
C*****
C***********************************************************************

      CALL SETUPF(.TRUE.,.FALSE.)
      CALL GTXS(TXTX,TXTY,10,'POLYMARKER')
      CALL FINSHF(.TRUE.)
      CALL GSCHH(0.03)
      CALL GTXS(0.1,0.68,15,'STANDARD SIZE :')
      CALL GTXS(0.1,0.43,19,'5 X STANDARD SIZE :')
      CALL GTXS(0.1,0.18,14,'MINIMAL SIZE :')

C***********************************************************************
C*****
C*****   Loop 5 times (once for each valid polymarker type) and
C*****   draw the polymarkers in varying colors and sizes.
C*****
C***********************************************************************

      DO 1000 I=1,5
        CALL GSMK(I)
        CALL GSMKSC(1.0)
        CALL GSPMCI(GWHITE)
        PMX(1) = (REAL(I) / 6.0)
        PMY(1) = 0.61
        CALL GPM(1,PMX,PMY)
        CALL GSMKSC(5.0)
        CALL GSPMCI(GLBLUE)
        PMY(1) = 0.32
        CALL GPM(1,PMX,PMY)
        CALL GSMKSC(1.0)
        CALL GSPMCI(GORNGE)
        PMY(1) = 0.13
        CALL GPM(1,PMX,PMY)
 1000 CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to put up a choice prompt indicating that lpfk
C*****   1 is return. RMENU will then call RCHOI to wait for choice
C*****   input and return when the 'return' choice is selected.
C*****
C***********************************************************************

      CALL RMENU(.TRUE.,CONT)
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: DEMOTX
C*****
C*****   Subroutine Function: This subroutne demonstrates the
C*****                        text output primitive and its various
C*****                        attributes: character expansion,
C*****                        character height, character spacing,
C*****                        character up vector, text color,
C*****                        text font and precision, text path,
C*****                        and text alignment.
C*****
C*****   Calls Subroutines:   SETUPF,FINSHF,RMENU
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE DEMOTX
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,ERRIND,I
      REAL*4       PX,PY,PAX(1),PAY(1),XCHU,YCHU
      REAL*4       XMIN,XMAX,YMIN,YMAX,CPX,CPY,TXEXPX(4)
      REAL*4       TXEXPY(4),RH,MINCHH
      LOGICAL      CONT
      CHARACTER    HEIGHT(16)*1
      CHARACTER*7  FONTNA(11)

      COMMON       /WINF/   WKID,WTYPE
      COMMON       /TEXT/   TXTX,TXTY
      COMMON       /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA  FONTNA /'FONT 1 ','FONT 2 ','FONT 3 ','FONT 4 ',
     *              'FONT 5 ','FONT 6 ','FONT 7 ','FONT 8 ',
     *              'FONT 9 ','FONT 10','FONT 11'/
      DATA  HEIGHT /'C','H','A','R','A','C','T','E','R',
     *              ' ','H','E','I','G','H','T'/

C***********************************************************************
C*****
C*****   Call subroutines SETUPF and FINSHF to draw the border and
C*****   border and title of the demo frame. Then draw the first
C*****   panel which says 'Select Text Attributes' in three
C*****   diagonal lines of different colors.
C*****
C***********************************************************************

      MINCHH = 0.002

      CALL SETUPF(.TRUE.,.FALSE.)
      CALL GTXS(TXTX,TXTY,4,'TEXT')
      CALL FINSHF(.TRUE.)

      CALL GSCHH(0.08)
      CALL GSTXFP(GFONT1,GSTRKP)
      CALL GSTXCI(GORNGE)
      CALL GSCHUP(-1.0,2.0)
      CALL GTXS(0.20,0.40,6,'SELECT')
      CALL GSCHXP(2.0)
      CALL GSTXCI(GLBLUE)
      CALL GTXS(0.08,0.16,4,'TEXT')
      CALL GSCHXP(1.0)
      CALL GSTXCI(GMGREN)
      CALL GTXS(0.32,0.10,10,'ATTRIBUTES')
      CALL GSCHUP(0.0,1.0)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate two prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

      CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the text panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

      IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

        CALL SETUPF(.FALSE.,.FALSE.)
        CALL GTXS(TXTX,TXTY,18,'FONT AND PRECISION')
        CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Draw a series of demo panels which show 11 of the text
C*****   fonts supported by GKS-CO. The panels will show font one
C*****   and two in the top half of the panel and then via segments,
C*****   change the bottom font each time the user selects the
C*****   continue choice key.
C*****
C***********************************************************************

        CALL GSCHH(0.1)
        CALL GSTXAL(GAHNOR,GAHALF)
        CALL GSTXCI(GRED)
        DO 1100 I=1,3
          CALL GCRSG(1001)
          CALL GSTXFP(I*3-2,GSTRKP)
          CALL GTXS(0.15,0.25,6,FONTNA(I*3-2))
          CALL GSTXFP(I*3-1,GSTRKP)
          CALL GTXS(0.15,0.50,6,FONTNA(I*3-1))
          CALL GSTXFP(I*3,GSTRKP)
          CALL GTXS(0.15,0.75,7,FONTNA(I*3))
          CALL GCLSG
          CALL RMENU(.FALSE.,CONT)
          CALL GDSG(1001)
	  CALL GUWK(WKID,1)
          IF (.NOT.(CONT)) GOTO 1110
 1100   CONTINUE
 1110   CONTINUE
	CALL GDSG(1259)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the text panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

        IF (CONT) THEN

C***********************************************************************
C*****
C*****   Reset the text alignment, font and precision.
C*****
C***********************************************************************

          CALL GSTXAL(GAHNOR,GAVNOR)
          CALL GSTXFP(GFONT1,GSTRKP)

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

          CALL SETUPF(.FALSE.,.FALSE.)
          CALL GTXS(TXTX,TXTY,6,'HEIGHT')
          CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Show how the character height can be varied by drawing
C*****   the letters 'character height', each in a different height.
C*****
C***********************************************************************

          PX = 0.025
          PY = 0.30
          RH = 3.0 * MINCHH
          CALL GSTXCI(GGREEN)
          CALL GSTXFP(GFONT1,GSTRKP)
          DO 1120 I=1,16
            CALL GSCHH(RH)
            CALL GTXS(PX,PY,1,HEIGHT(I))
            CALL GQTXXS(WKID,PX,PY,1,HEIGHT(I),ERRIND,CPX,CPY,
     *                  TXEXPX,TXEXPY)
            PX = CPX
            RH = RH + 0.0085
 1120     CONTINUE

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

          CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the text panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

          IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

            CALL SETUPF(.FALSE.,.FALSE.)
            CALL GTXS(TXTX,TXTY,9,'UP-VECTOR')
            CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Show how the character up-vector can be varied by drawing
C*****   the word 'GKS' in a circle with each word a different
C*****   color and in a different 'up' position.
C*****
C***********************************************************************

            CALL GSWN(1,-0.050,1.050,-0.020,1.080)
            PX = 0.58
            PY = 0.05
            XCHU = -0.5
            YCHU = 1.0
            CALL GSCHH(0.04)
            CALL GSTXFP(GFONT1,GSTRKP)
            DO 1130 I=1,16
              IF (I.EQ.16) THEN
                CALL GSTXCI(I-1)
              ELSE
                CALL GSTXCI(I)
              ENDIF
              CALL GSCHUP(XCHU,YCHU)
              CALL GTXS(PX,PY,5,' GKS ')
              CALL GQTXXS(WKID,PX,PY,5,' GKS ',ERRIND,CPX,CPY,
     *                    TXEXPX,TXEXPY)
              PX = CPX
              PY = CPY
              IF ((I.GE.2).AND.(I.LE.5)) THEN
                YCHU = YCHU - 0.5
              ELSE
                IF ((I.GE.6).AND.(I.LE.9)) THEN
                  XCHU = XCHU + 0.5
                ELSE
                  IF ((I.GE.10).AND.(I.LE.13)) THEN
                    YCHU = YCHU + 0.5
                  ELSE
                    XCHU = XCHU - 0.5
                  ENDIF
                ENDIF
              ENDIF
 1130       CONTINUE
            CALL GSCHUP(0.0,1.0)
            CALL GSWN(1,0.00,1.00,0.00,1.00)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

            CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the text panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

            IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

              CALL SETUPF(.FALSE.,.FALSE.)
              CALL GTXS(TXTX,TXTY,9,'EXPANSION')
              CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Show the effects of the character expansion attribute by
C*****   varying the expansion factor and writing out the word
C*****   'expansion' four times.
C*****
C***********************************************************************

              PX = 0.03
              PY = 0.75
              CALL GSCHH(0.1)
              CALL GSTXCI(GORNGE)
              CALL GSTXFP(GFONT1,GSTRKP)
              CALL GSVP(1,0.015,0.985,0.015,0.900)
              DO 1140 I=1,4
                CALL GSCHXP(0.2 + REAL(I) * 0.30)
                CALL GTXS(PX,PY,9,'EXPANSION')
                PY = PY - 0.2
 1140         CONTINUE
              CALL GSCHXP(1.0)
              CALL GSVP(1,0.0,1.0,0.0,1.0)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

              CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the text panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

              IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

                CALL SETUPF(.FALSE.,.FALSE.)
                CALL GTXS(TXTX,TXTY,4,'PATH')
                CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Show the effects of the character path attribute by
C*****   drawing the words 'right', 'left', 'up', and 'down'
C*****   with their corresponding paths set.
C*****
C***********************************************************************

                PX = 0.48
                PY = 0.50
                CALL GSTXCI(GWHITE)
                CALL GSTXFP(GFONT1,GSTRKP)
                CALL GSCHH(0.08)
                CALL GTXS(PX,PY,6,' RIGHT')
                CALL GSTXP(GLEFT)
                CALL GTXS(PX,PY,5,' LEFT')
                CALL GSTXP(GUP)
                CALL GTXS(PX,PY,3,' UP')
                CALL GSTXP(GDOWN)
                CALL GTXS(PX,PY,4,'DOWN')
                CALL GSTXP(GRIGHT)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

                CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the text panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

                IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

                  CALL SETUPF(.FALSE.,.FALSE.)
                  CALL GTXS(TXTX,TXTY,9,'ALIGNMENT')
                  CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Show the effects of the text alignment attribute by
C*****   displaying words that correspond with the various
C*****   alignment settings.
C*****
C***********************************************************************

                  CALL GSWN(1,-0.020,1.080,-0.050,1.050)
                  PAX(1) = 0.02
                  PAY(1) = 0.720
                  CALL GSCHH(0.05)
                  CALL GSTXFP(GFONT1,GSTRKP)
                  CALL GSTXCI(GMAGNT)
                  CALL GSPMCI(GWHITE)
                  DO 1150 I=1,5
                    CALL GPM(1,PAX,PAY)
                    CALL GSTXAL(GAHNOR,I)
                    IF (I.EQ.1) THEN
                      CALL GTXS(PAX,PAY,4,'TOP ')
                      CALL GQTXXS(WKID,PAX,PAY,4,'TOP ',ERRIND,
     *                            CPX,CPY,TXEXPX,TXEXPY)
                    ELSE
                      IF (I.EQ.2) THEN
                        CALL GTXS(PAX,PAY,4,'CAP ')
                        CALL GQTXXS(WKID,PAX,PAY,4,'CAP ',ERRIND,
     *                              CPX,CPY,TXEXPX,TXEXPY)
                      ELSE
                        IF (I.EQ.3) THEN
                          CALL GTXS(PAX,PAY,5,'HALF ')
                          CALL GQTXXS(WKID,PAX,PAY,5,'HALF ',ERRIND,
     *                                CPX,CPY,TXEXPX,TXEXPY)
                        ELSE
                          IF (I.EQ.4) THEN
                            CALL GTXS(PAX,PAY,5,'BASE ')
                            CALL GQTXXS(WKID,PAX,PAY,5,'HALF ',ERRIND,
     *                                  CPX,CPY,TXEXPX,TXEXPY)
                          ELSE
                            IF (I.EQ.5) THEN
                              CALL GTXS(PAX,PAY,6,'BOTTOM')
                            ENDIF
                          ENDIF
                        ENDIF
                      ENDIF
                    ENDIF
                    PAX(1) = CPX
 1150             CONTINUE
                  PAX(1) = 0.50
                  PAY(1) = 0.40
                  CALL GSCHH(0.1)
                  CALL GSTXAL(GACENT,GAVNOR)
                  CALL GPM(1,PAX,PAY)
                  CALL GTXS(PAX,PAY,6,'CENTER')
                  PAY(1) = 0.25
                  CALL GSTXAL(GARITE,GAVNOR)
                  CALL GPM(1,PAX,PAY)
                  CALL GTXS(PAX,PAY,5,'RIGHT')
                  PAY(1) = 0.10
                  CALL GSCHH(0.1)
                  CALL GSTXAL(GALEFT,GAVNOR)
                  CALL GPM(1,PAX,PAY)
                  CALL GTXS(PAX,PAY,4,'LEFT')
                  CALL GSTXAL(GAHNOR,GAVNOR)
                  CALL GSWN(1,0.00,1.00,0.00,1.00)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompts
C*****   for the lpfks (next and return). RMENU will call RCHOI
C*****   to wait for a valid choice selection and will pass that
C*****   selection back in CHNR.
C*****
C***********************************************************************

                  CALL RMENU(.FALSE.,CONT)

C***********************************************************************
C*****
C*****   If user chose to continue viewing the text panels,
C*****   continue on. Otherwise, fall out the bottom and return
C*****   to the output primitive screen.
C*****
C***********************************************************************

                  IF (CONT) THEN

C***********************************************************************
C*****
C*****   Call SETUPF and FINSHF to draw the border and title of
C*****   the demo frame.
C*****
C***********************************************************************

                    CALL SETUPF(.FALSE.,.FALSE.)
                    CALL GTXS(TXTX,TXTY,7,'SPACING')
                    CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Show the effects of the character spacing attribute by
C*****   vaying the spacing factor and writing out the sentence
C*****   'GKS - OH WHAT A MESS'.
C*****
C***********************************************************************

                    PX = 0.03
                    PY = 0.75
                    CALL GSTXFP(GFONT1,GSTRKP)
                    CALL GSCHH(0.07)
                    CALL GSTXCI(GLGREN)
                    CALL GSVP(1,0.015,0.985,0.015,0.900)
                    DO 1160 I=1,4
                      CALL GSCHSP(-0.8 + REAL(I) * 0.3)
                      CALL GTXS(PX,PY,16,'GKS - SPACED OUT')
                      PY = PY - 0.2
 1160               CONTINUE
                    CALL GSCHSP(0.0)
                    CALL GSVP(1,0.0,1.0,0.0,1.0)

C***********************************************************************
C*****
C*****   Call RMENU to use pack data record to indicate to prompt
C*****   for the lpfk (return). RMENU will call RCHOI to wait for
C*****   a valid choice selection and will pass that selection back
C*****   in CHNR. But in this case, the user could only select
C*****   return as a valid choice, so we will fall down to the
C*****   return statement and go back to the output primitves.
C*****
C***********************************************************************

                    CALL RMENU(.TRUE.,CONT)
                  ENDIF
                ENDIF
              ENDIF
            ENDIF
          ENDIF
        ENDIF
      ENDIF

      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: DRWHLS
C*****
C*****   Subroutine Function: Draw a the wheels for the car picture
C*****                        through this common subroutine.
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE DRWHLS(MITX,MITY,N,R)

      INTEGER*4    N,I
      REAL*4       MITX,MITY,R,ALPHA,RX(21),RY(21)

      ALPHA = 0.0
      DO 1200 I=1,N
        RX(I) = MITX + R * SIN(ALPHA)
        RY(I) = MITY + R * COS(ALPHA)
        ALPHA = ALPHA + 2.0 * 3.141593 / REAL(N)
 1200 CONTINUE
      RX(N+1) = RX(1)
      RY(N+1) = RY(1)
      CALL GPL(N+1,RX,RY)

      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: FINSHF
C*****
C*****   Subroutine Function: Finish the border border for each
C*****                        screen of the demo.
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE FINSHF(MFRAME)
      INCLUDE 'gkspar.inc'

      REAL*4     PX2(5),PY3(5),PY5(5)
      LOGICAL    MFRAME

      DATA PX2 /0.015,0.015,0.985,0.985,0.015/
      DATA PY3 /0.015,0.750,0.750,0.015,0.015/
      DATA PY5 /0.015,0.900,0.900,0.015,0.015/

      IF (MFRAME) THEN
        CALL GPL (5,PX2,PY3)
      ELSE
        CALL GPL(5,PX2,PY5)
      ENDIF
      CALL GSTXFP(GFONT1,GSTRKP)
      CALL GSTXAL(GAHNOR,GAVNOR)
      CALL GCLSG
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: INTCRE
C*****
C*****   Subroutine Function: This subroutine will create a new
C*****                        segment. The user can choose to
C*****                        create a segment with a polyline,
C*****                        polymarker, fill area or text in it.
C*****
C*****   Calls Subroutines:   RCHOI,CREPLN,CREPMK,CREFA,CRETXT
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE INTCRE
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,CHNR,ERRIND
      INTEGER*4    NELVAL,LCHVAL(1),LDRVAL
      INTEGER*4    NELPRM,LCHPRM(5),LDRPRM

      REAL*4       XMIN,XMAX,YMIN,YMAX

      CHARACTER*1  MENVAL(1)
      CHARACTER*80 DTRVAL(1)
      CHARACTER*12 MENPRM(5)
      CHARACTER*80 DTRPRM(5)

      COMMON      /WINF/   WKID,WTYPE
      COMMON      /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA MENVAL /' '/
      DATA LCHVAL /1/
      DATA NELVAL /1/

      DATA MENPRM /'POLYLINE','POLYMARKER','FILL AREA','TEXT','RETURN'/
      DATA LCHPRM /8,10,9,4,6/
      DATA NELPRM /5/

C***********************************************************************
C*****
C*****   Initialize the valuator for scale factors requested
C*****   later. Initialize the choice device and call RCHOI to
C*****   wait for the user to select which type of segment it
C*****   wishes to create: polyline, polymarker, fill area or
C*****   text. Call the appropriate subroutine based on that
C*****   selection or exit (choice number = 5).
C*****
C***********************************************************************

      CALL GPREC(0,0,0,0.0,NELVAL,LCHVAL,MENVAL,NELVAL,ERRIND,
     *           LDRVAL,DTRVAL)
      CALL GINVL(WKID,1,1.0,1,((XMAX-XMIN) * 0.66 + XMIN),XMAX,
     *           YMIN,YMAX,0.0,10.0,LDRVAL,DTRVAL)

      CALL GPREC(0,0,0,0.0,NELPRM,LCHPRM,MENPRM,NELPRM,
     *           ERRIND,LDRPRM,DTRPRM)
 1300 CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LDRPRM,DTRPRM)
      CALL GMSGS(WKID,23,'SELECT OUTPUT PRIMITIVE')
      CALL RCHOI(WKID,5,CHNR)

C***********************************************************************
C*****
C*****   Call the appropriate subroutine based on the choice
C*****   returned from the user. If choice was RETURN (lpfk 5),
C*****   exit back to main interactive menu panel.
C*****
C***********************************************************************

      IF (CHNR.EQ.1) THEN
        CALL CREPLN
      ELSE
        IF (CHNR.EQ.2) THEN
          CALL CREPMK
        ELSE
          IF (CHNR.EQ.3) THEN
            CALL CREFA
          ELSE
            IF (CHNR.EQ.4) THEN
              CALL CRETXT
            ELSE
              GOTO 1310
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      GOTO 1300
 1310 RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: INTDEL
C*****
C*****   Subroutine Function:  Deletes a segment which is picked
C*****                         by the user with the locator
C*****                         device.
C*****
C*****   Calls Subroutine:     CRSEGM
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE INTDEL
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,STAT,SGNA,PCID

      LOGICAL      SGNAME(50)

      COMMON      /WINF/   WKID,WTYPE
C The AIX fortran compiler only allows common blocks to be
C dimensioned once. The following line was modifed for AIX port. c2071
      COMMON      /SEGM/   SGNAME

C***********************************************************************
C*****
C*****   Call subroutine CRSEGM to create a segment and draw
C*****   a red box with 'RETURN' in it in the upper left hand
C*****   corner of the screen. This segment can be picked by
C*****   the user when it wants to return to the previous menu
C*****   frame (segment number = 6). If a valid segment was
C*****   picked, either delete the segment and wait for the
C*****   next input, or return.
C*****
C***********************************************************************

      CALL CRSEGM
 1400 CALL GMSGS(WKID,32,'PICK SEGMENT TO DELETE OR RETURN')
 1410 CALL GRQPK(WKID,1,STAT,SGNA,PCID)
      IF (STAT.EQ.1) THEN
        IF (SGNA.GE.7) THEN
          SGNAME(SGNA) = .FALSE.
          CALL GDSG(SGNA)
          GOTO 1400
        ELSE
          CALL GDSG(6)
        ENDIF
      ELSE
        GOTO 1410
      ENDIF
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: INTER
C*****
C*****   Subroutine Function: This subroutine will demonstrate the
C*****                        use of GKS in doing interactive
C*****                        graphics functions. It will also use
C*****                        the valuator and stroke input devices
C*****                        for the first time.
C*****
C*****   Calls Subroutines:   RCHOI,INTCRE,INTINS,INTTRA,INTDEL
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE INTER
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,WISS,CHNR,NUMELA,LCHSTR(5),LENGTH,ERRIND

      REAL*4       FRAME(10),FA3(6),FA4(8),FA5(10),XMIN,XMAX,YMIN,YMAX

      CHARACTER*12 MENU(5)
      CHARACTER*80 DTREC(5)

      LOGICAL      SGNAME(50)

      COMMON      /WINF/   WKID,WTYPE
C The AIX fortran compiler only allows common blocks to be
C dimensioned once. The following line was modifed for AIX port. c2071
      COMMON      /SEGM/   SGNAME
      COMMON      /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA MENU   /'CREATE','INSERT','TRANSFORM','DELETE','RETURN'/
      DATA LCHSTR /6,6,9,6,6/
      DATA NUMELA /5/

      DATA FRAME  /0.000,0.000,1.000,1.000,0.000,
     *             0.000,1.000,1.000,0.000,0.000/
      DATA FA3    /0.400,0.600,0.600,0.400,0.400,0.800/
      DATA FA4    /0.100,0.200,0.300,0.200,0.200,0.100,0.200,0.300/
      DATA FA5    /0.300,0.400,0.350,0.250,0.200,
     *             0.600,0.500,0.400,0.400,0.500/

C***********************************************************************
C*****
C*****   Open the WISS workstation. Initialize the list of segments
C*****   available (SGNAME) to available (false). The first six are
C*****   used by GKSDEMO and therefore never checked for their
C*****   availibility.
C*****
C***********************************************************************

      WISS = WKID + 1
      CALL GOPWK(WISS,1,3)
      CALL GACWK(WISS)
      DO 1500 I=7,50
        SGNAME(I) = .FALSE.
 1500 CONTINUE

C***********************************************************************
C*****
C*****   Clear the workstation and draw the initial screen for the
C*****   interactive portion of the demo. This screen consists of
C*****   three segments: one containing a red, hollow triangle. A
C*****   second segment containing a blue, hatched pentagon. And a
C*****   third segment containing a green, solid square.
C*****
C***********************************************************************

      CALL GCLRWK(WKID,1)
      CALL GSLN(GLSOLI)
      CALL GSLWSC(1.0)
      CALL GSPLCI(GYELOW)
      CALL GPL(5,FRAME(1),FRAME(6))
      CALL GCRSG(7)
      CALL GSDTEC(7,GDETEC)
      CALL GSFAIS(GHOLLO)
      CALL GSFACI(GRED)
      CALL GFA(3,FA3(1),FA3(4))
      CALL GCLSG
      SGNAME(7) = .TRUE.
      CALL GCRSG(8)
      CALL GSDTEC(8,GDETEC)
      CALL GSFAIS(GHATCH)
      CALL GSFACI(GBLUE)
      CALL GFA(5,FA5(1),FA5(6))
      CALL GCLSG
      SGNAME(8) = .TRUE.
      CALL GCRSG(9)
      CALL GSDTEC(9,GDETEC)
      CALL GSFAIS(GSOLID)
      CALL GSFACI(GGREEN)
      CALL GFA(4,FA4(1),FA4(5))
      CALL GCLSG
      SGNAME(9) = .TRUE.

C***********************************************************************
C*****
C*****   Initialize the choice device and call subroutine RCHOI to
C*****   wait for one of the following choices:
C*****        Choice 1: Create Segment
C*****               2: Insert Segment
C*****               3: Transform Segment
C*****               4: Delete Segment
C*****               5: Return
C*****   Call the appropriate function depending on the input
C*****   received from the choice device, or exit.
C*****
C***********************************************************************

      CALL GPREC(0,0,0,0.0,NUMELA,LCHSTR,MENU,NUMELA,
     *           ERRIND,LENGTH,DTREC)
 1510 CALL GMSGS(WKID,1,' ')
      CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LENGTH,DTREC)
      CALL RCHOI(WKID,5,CHNR)
      IF (CHNR.EQ.1) THEN
        CALL INTCRE
      ELSE
        IF (CHNR.EQ.2) THEN
          CALL INTINS
        ELSE
          IF (CHNR.EQ.3) THEN
            CALL INTTRA
          ELSE
            IF (CHNR.EQ.4) THEN
              CALL INTDEL
            ELSE
              GOTO 1520
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      GOTO 1510
1520  CALL GDAWK(WISS)
      CALL GCLWK(WISS)
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: INTINS
C*****
C*****   Subroutine Function:  This subroutine will insert chosen
C*****                         segments into a new segment and
C*****                         delete the old segment(s).
C*****
C*****   Calls Subroutines:    CRSEGM
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE INTINS
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,I,STAT,SGNA,PCID

      REAL*4       TM(2,3)

      LOGICAL      SGNAME(50)

      COMMON      /WINF/   WKID,WTYPE
C The AIX fortran compiler only allows common blocks to be
C dimensioned once. The following line was modifed for AIX port. c2071
      COMMON      /SEGM/   SGNAME

C***********************************************************************
C*****
C*****   Set up a transformation matrix to hold new segment data.
C*****   Loop through list of segment numbers (segments 1 - 6 are
C*****   reserved for GKSDEMO use) and find the first free segment.
C*****
C***********************************************************************

      CALL GEVTM(0.0,0.0,0.0,0.0,0.0,1.0,1.0,1,TM)
      DO 1600 I=7,50
        IF (.NOT.(SGNAME(I))) THEN
          GOTO 1610
        ENDIF
 1600 CONTINUE
      CALL GMSGS(WKID,32,'NO MORE THAN 50 SEGMENTS ALLOWED')
      GOTO 1640
 1610 CALL CRSEGM

C***********************************************************************
C*****
C*****   Create a new segment and wait for the user to choose all
C*****   existing segments it wants added to the new segment.
C*****
C***********************************************************************

      SGNAME(I) = .TRUE.
      CALL GCRSG(I)
      CALL GSDTEC(I,1)
      CALL GSVIS(I,1)
 1620 CALL GMSGS(WKID,32,'PICK SEGMENT TO INSERT OR RETURN')
 1630 CALL GRQPK(WKID,1,STAT,SGNA,PCID)
      IF (STAT.EQ.1) THEN

C***********************************************************************
C*****
C*****   If user picked an existing segment for inserting, insert
C*****   the segment in the new segment and delete the old one.
C*****   Continue loop until user chooses 'return' (segment 6).
C*****
C***********************************************************************

        IF (SGNA.GE.7) THEN
          CALL GINSG(SGNA,TM)
          CALL GDSG(SGNA)
          SGNAME(SGNA) = .FALSE.
          GOTO 1620
        ELSE
          CALL GCLSG
          CALL GDSG(6)
          GOTO 1640
        ENDIF
      ELSE
        GOTO 1630
      ENDIF
 1640 RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: INTTRA
C*****
C*****   Subroutine Function:  Perform transformations on selected
C*****                         segments, allowing the user to
C*****                         input the various transformation
C*****                         variables: fixed point, shift
C*****                         factor, rotation angle and x,y
C*****                         scaling factors.
C*****
C*****   Calls Subroutine:    RCHOI
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE INTTRA
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,CHNR1,ERRIND,SW
      INTEGER*4    STAT,SGNA,PCID,NUMEL1,LCHST1(6),LENGT1,STATVL
      INTEGER*4    VIS,HIGHL,DET,NUMEL2,LCHST2(1),LENGT2,STATLC,TNR

      REAL*4       XMIN,XMAX,YMIN,YMAX
      REAL*4       SEGTM(2,3),SGPR,X0,Y0,DX,DY,PHI,FX,FY
      REAL*4       MOUT(2,3),PX1(1),PY1(1),SCFACT,PX2(1),PY2(1)

      CHARACTER*12 MENU1(6)
      CHARACTER*1  MENU2(1)
      CHARACTER*80 DTREC1(6)
      CHARACTER*80 DTREC2(1)

      COMMON      /WINF/   WKID,WTYPE
      COMMON      /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA MENU1  /'TRANS-SGMT','FIXPOINT','SHIFT','ROTATE','SCALING',
     1             'RETURN'/
      DATA LCHST1 /10,8,5,6,7,6/
      DATA NUMEL1 /6/

      DATA MENU2  /' '/
      DATA LCHST2 /1/
      DATA NUMEL2 /1/

C***********************************************************************
C*****
C*****   Set up the transformation default values. Use pack data
C*****   record to set up the prompts for the choice device. The
C*****   choices are: transformation segment, fixed point, shift,
C*****   rotate and scaling factors, and return.
C*****
C***********************************************************************

      X0 = 0.5
      Y0 = 0.5
      DX = 0.0
      DY = 0.0
      PHI = 0.0
      FX = 1.0
      FY = 1.0
      SW = 0
      CALL GSMK(GAST)
      CALL GSMKSC(1.0)
      CALL GSPMCI(GBLUE)
      CALL GPREC(0,0,0,0.0,NUMEL1,LCHST1,MENU1,NUMEL1,
     *           ERRIND,LENGT1,DTREC1)
      CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LENGT1,DTREC1)
 1700 CALL GMSGS(WKID,26,' SELECT FOR TRANSFORMATION')
      CALL RCHOI(WKID,6,CHNR1)

C***********************************************************************
C*****
C*****   If the user chose to pick the transformation segment,
C*****   then call subroutine to add a pickable segment for
C*****   'RETURN' (segment 6). Read the segment number that was
C*****   picked. If not the return segment, then perform the
C*****   transformation with the current values in each of the
C*****   transformation parameters. Otherwise, delete segment 6
C*****   (return segment) and exit the transformation routine.
C*****
C***********************************************************************

      IF (CHNR1.EQ.1) THEN
        CALL CRSEGM
        CALL GMSGS(WKID,22,'PICK SEGMENT OR RETURN')
 1710   CALL GRQPK(WKID,1,STAT,SGNA,PCID)
        IF (STAT.EQ.1) THEN
          IF (SGNA.GE.7) THEN
            CALL GQSGA(SGNA,ERRIND,SEGTM,VIS,HIGHL,SGPR,DET)
            CALL GACTM(SEGTM,X0,Y0,DX,DY,PHI,FX,FY,SW,MOUT)
            CALL GSSGT(SGNA,MOUT)
            CALL GMSGS(WKID,27,'PICK NEXT SEGMENT OR RETURN')
            GOTO 1710
          ELSE
            CALL GDSG(6)
          ENDIF
        ENDIF
      ELSE

C***********************************************************************
C*****
C*****   Set up the pick device so that the user can enter a
C*****   fixed point using the locator device. Once a fixed point
C*****   has been returned, exit back to the main choice panel.
C*****
C***********************************************************************

        IF (CHNR1.EQ.2) THEN
          CALL GPREC(0,0,0,0.0,NUMEL2,LCHST2,MENU2,NUMEL2,
     *               ERRIND,LENGT2,DTREC2)
          CALL GINLC(WKID,1,1,0.1,0.1,2,XMIN,XMAX,YMIN,YMAX,
     *               LENGT2,DTREC2)
          CALL GMSGS(WKID,17,'LOCATE FIX POINT')
 1720     CALL GRQLC(WKID,1,STATLC,TNR,PX1(1),PY1(1))
          IF (STATLC.EQ.1) THEN
            X0 = PX1(1)
            Y0 = PY1(1)
          ELSE
            GOTO 1720
          ENDIF
        ELSE

C***********************************************************************
C*****
C*****   Set up the pick device so that the user can enter two
C*****   point from which the shift factor can be determined.
C*****   Once both points have been entered, calculate the shift
C*****   factors and return to the main choice panel.
C*****
C***********************************************************************

          IF (CHNR1.EQ.3) THEN
            CALL GINLC(WKID,1,1,0.1,0.1,3,XMIN,XMAX,YMIN,YMAX,
     *                 LENGT2,DTREC2)
            CALL GMSGS(WKID,18,'LOCATE FIRST POINT')
 1730       CALL GRQLC(WKID,1,STATLC,TNR,PX1(1),PY1(1))
            IF (STATLC.EQ.1) THEN
              CALL GPREC(1,0,0,0.0,0,LCHST2,MENU2,NUMEL2,
     *                   ERRIND,LENGT2,DTREC2)
              CALL GINLC(WKID,1,1,PX1(1),PY1(1),4,XMIN,XMAX,
     *                   YMIN,YMAX,LENGT2,DTREC2)
              CALL GMSGS(WKID,19,'LOCATE SECOND POINT')
 1740         CALL GRQLC(WKID,1,STATLC,TNR,PX2(1),PY2(1))
              IF (STATLC.EQ.1) THEN
                DX = PX2(1) - PX1(1)
                DY = PY2(1) - PY1(1)
              ELSE
                GOTO 1740
              ENDIF
            ELSE
              GOTO 1730
            ENDIF
          ELSE

C***********************************************************************
C*****
C*****   Use the valuator input to determine the rotation angle in
C*****   radians. Once input from the valuator has been received
C*****   (user turns valuator and then hits ENTER), return back to
C*****   the main transformation choice panel.
C*****
C***********************************************************************

            IF (CHNR1.EQ.4) THEN
              CALL GINVL(WKID,1,1.0,1,((XMAX-XMIN) * 0.66 + XMIN),
     *                   XMAX,YMIN,YMAX,0.0,6.2832,LENGT2,DTREC2)
              CALL GMSGS(WKID,35,
     *                   'EVALUATE ROTATION ANGLE IN RADIANS')
 1750         CALL GRQVL(WKID,1,STATVL,SCFACT)
              IF (STATVL.EQ.1) THEN
                PHI = SCFACT
              ELSE
                GOTO 1750
              ENDIF
            ELSE

C***********************************************************************
C*****
C*****   Use the valuator input to determine the x and y scaling
C*****   factors. Once both values have been received (user turns
C*****   valuator and then hits ENTER), return back to the main
C*****   transformation choice panel.
C*****
C***********************************************************************

              IF (CHNR1.EQ.5) THEN
                CALL GINVL(WKID,1,1.0,1,
     *                     ((XMAX-XMIN) * 0.66 + XMIN),XMAX,
     *                     YMIN,YMAX,0.0,5.0,LENGT2,DTREC2)
                CALL GMSGS(WKID,25,'EVALUATE X - SCALE FACTOR')
 1760           CALL GRQVL(WKID,1,STATVL,SCFACT)
                IF (STATVL.EQ.1) THEN
                  FX = SCFACT
                  CALL GMSGS(WKID,25,'EVALUATE Y - SCALE FACTOR')
 1770             CALL GRQVL(WKID,1,STATVL,SCFACT)
                  IF (STATVL.EQ.1) THEN
                    FY = SCFACT
                  ELSE
                    GOTO 1770
                  ENDIF
                ELSE
                  GOTO 1760
                ENDIF

C***********************************************************************
C*****
C*****   User selected choice key 6 - RETURN. Exit out of the
C*****   transformation panel and back to the Interactive main
C*****   choice panel.
C*****
C***********************************************************************

              ELSE
                GOTO 1780
              ENDIF
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      GOTO 1700
 1780 RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: MAPDEM
C*****
C*****   Subroutine Function: High level picture of how the GKS-CO
C*****                        mapper fits into the graPHIGS systems
C*****                        picture.
C*****
C*****   Calls Subroutines:   SETUPF,FINSHF,RMENU
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE MAPDEM
      INCLUDE 'gkspar.inc'

      INTEGER*4 WKID,WTYPE
      REAL*4    GKSX(4),PAPPX(4),GAPPX(4),PHIGSX(4),PWKX(4),
     *          GKSY(4),PAPPY(4),GAPPY(4),PHIGSY(4),PWKY(4)
      REAL*4    PLX(27),PLY(27),ALPHA
      REAL*4    XMIN,XMAX,YMIN,YMAX,TXTX,TXTY
      REAL*4    PAPX(10),GAGX(10),GPX(10),PWX(10),
     *          PAPY(10),GAGY(10),GPY(10),PWY(10),
     *          PLATTX(10),PLATTY(10),X(10),WKMX(3),WKMY(3)
      LOGICAL   CONT

      COMMON    /WINF/   WKID,WTYPE
      COMMON    /LIMITS/ XMIN,XMAX,YMIN,YMAX
      COMMON    /TEXT/ TXTX,TXTY

      DATA    EMPTRP /0.0/
      DATA    GKSX   /08.3,08.3,14.5,14.5/,
     *        GKSY   /09.5,13.5,13.5,09.5/
      DATA    PAPPX  /01.5,01.5,07.7,07.7/,
     *        PAPPY  /14.5,15.8,15.8,14.5/
      DATA    GAPPX  /08.3,08.3,14.5,14.5/,
     *        GAPPY  /14.5,15.8,15.8,14.5/
      DATA    PHIGSX /01.5,01.5,14.5,14.5/,
     *        PHIGSY /04.3,08.8,08.8,04.3/
      DATA    PWKX   /01.5,01.5,03.5,03.5/,
     *        PWKY   /01.2,03.5,03.5,01.2/
      DATA    PAPX   /04.6,04.5,04.6,04.7,04.6,04.6,04.5,04.6,04.7,04.6/
     *        PAPY   /14.3,14.3,14.5,14.3,14.3,09.0,09.0,08.8,09.0,09.0/
      DATA    GAGX   /11.4,11.3,11.4,11.5,11.4,11.4,11.3,11.4,11.5,11.4/
     *        GAGY   /14.3,14.3,14.5,14.3,14.3,13.7,13.7,13.5,13.7,13.7/
      DATA    GPX    /11.4,11.3,11.4,11.5,11.4,11.4,11.3,11.4,11.5,11.4/
     *        GPY    /09.3,09.3,09.5,09.3,09.3,09.0,09.0,08.8,09.0,09.0/
      DATA    PWX    /02.5,02.4,02.5,02.6,02.5,02.5,02.4,02.5,02.6,02.5/
     *        PWY    /04.1,04.1,04.3,04.1,04.1,03.7,03.7,03.5,03.7,03.7/
      DATA    PLATTX /14.7,14.7,14.5,14.7,14.7,15.1,15.1,15.3,15.1,15.1/
     *        PLATTY /12.5,12.6,12.5,12.4,12.5,12.5,12.6,12.5,12.4,12.5/
      DATA    WKMX   /08.0,09.5,11.0/,
     *        WKMY   /2.35,2.35,2.35/

C***********************************************************************
C*****
C*****   Call the setup frame and finish frame subroutines to
C*****   create the border and title of the display frame for
C*****   the gks-co diagram of the demo. This frame of the
C*****   demo shows a flowchart type picture of how gks-co fits
C*****   in with graPHIGS and both gks and phigs applications.
C*****
C***********************************************************************

      CALL SETUPF(.FALSE.,.FALSE.)
      CALL GTXS(TXTX,TXTY,20,'        XGKS        ')
      CALL FINSHF(.FALSE.)

C***********************************************************************
C*****
C*****   Set the window, view and some text and fill area
C*****   parameters.
C*****
C***********************************************************************

      CALL GSCHXP(0.9)
      CALL GSTXFP(GFONT1,GSTRKP)
      CALL GSWN(2,0.0,18.0,0.0,18.0)
      CALL GSVP(2,0.0,1.0,0.0,1.0)
      CALL GSELNT(2)
      CALL GSFAIS(1)

C***********************************************************************
C*****
C*****   Create a blue rectangle to hold the text which will
C*****   denote the graPHIGS API layer in the flowchart picture.
C*****
C***********************************************************************

      CALL GSFACI(GBLUE)
      CALL GFA(4,PHIGSX,PHIGSY)
      CALL GSCHH(1.0)
      CALL GSTXAL(GACENT,GAHALF)
      CALL GSTXCI(GWHITE)
      CALL GSTXFP(GFONT2,GSTRKP)
      CALL GSCHSP(0.0)
      CALL GTXS(8.0,6.55,17,'graPHIGS (TM) API')

C***********************************************************************
C*****
C*****   Create three small light blue rectangles at the bottom
C*****   of the flowchart to denote the workstations connected
C*****   via graPHIGS.
C*****
C***********************************************************************

      CALL GSFACI(GLBLUE)
      CALL GSTXCI(GBLUE)
      CALL GSCHH(1.0)
      CALL GSCHSP(0.0)
      CALL GFA(4,PWKX,PWKY)
      CALL GTXS(2.5,2.35,1,'1')
      DO 1800 I=1,4
        X(I) = PWKX(I) + 3.0
 1800 CONTINUE
      CALL GFA(4,X,PWKY)
      CALL GTXS(5.5,2.35,1,'2')
      DO 1810 I=1,4
        X(I) = X(I) + 8.0
 1810 CONTINUE
      CALL GFA(4,X,PWKY)
      CALL GTXS(13.5,2.35,1,'N')
      CALL GSTXCI(GWHITE)
      CALL GSCHH(0.6)
      CALL GSCHSP(0.0)
      CALL GSCHXP(1.2)
      CALL GSTXAL(GACENT,GAVNOR)
      CALL GTXS(8.0,0.35,12,'WORKSTATIONS')

C***********************************************************************
C*****
C*****   Create an orange rectangle above the graPHIGS API layer
C*****   to denote the GKS-CO interpreter layer.
C*****
C***********************************************************************

      CALL GSCHSP(0.0)
      CALL GSCHXP(0.9)
      CALL GSFACI(GORNGE)
      CALL GFA(4,GKSX,GKSY)
      CALL GSCHH(0.9)
      CALL GSTXAL(GACENT,GAHALF)
      CALL GTXS(11.4,11.5,6,'GKS-CO')

C***********************************************************************
C*****
C*****   Write Metafile Input Output directly under the picture of
C*****   a disk storage device (this will be drawn later in the
C*****   program).
C*****
C***********************************************************************

      CALL GSCHH(0.2)
      CALL GSCHSP(0.0)
      CALL GSTXFP(GFONT1,GSTRKP)
      CALL GTXS(16.3,11.5,8,'METAFILE')
      CALL GTXS(16.3,11.2,5,'INPUT')
      CALL GTXS(16.3,10.9,6,'OUTPUT')

C***********************************************************************
C*****
C*****   Create two green rectangles at the top of the flowchart,
C*****   one denoting a phigs application and one denoting a gks
C*****   application.
C*****
C***********************************************************************

      CALL GSCHSP(0.0)
      CALL GSFACI(GMGREN)
      CALL GSTXCI(GWHITE)
      CALL GSCHSP(0.0)
      CALL GFA(4,GAPPX,GAPPY)
      CALL GSCHH(0.3)
      CALL GTXS(11.4,15.15,15,'GKS APPLICATION')
      CALL GSCHH(0.25)
      CALL GFA(4,PAPPX,PAPPY)
      CALL GTXS(4.6,15.15,25,'graPHIGS (TM) APPLICATION')

C***********************************************************************
C*****
C*****   Calculate the points in order to draw a grey-blue disk
C*****   storage device off to the side of the GKS-CO layer that
C*****   will denote the GKS-CO metafile capability.
C*****
C***********************************************************************

      CALL GSTXCI(GBLUE)
      CALL GSCHSP(0.0)
      PLX(27) = 16.3
      PLY(27) = 13.0
      ALPHA = -3.141593 / 2.0
      DO 1820 I = 1,13
        PLX(I) = PLX(27) + 1.00 * SIN(ALPHA)
        PLY(I) = PLY(27) + 0.25 * COS(ALPHA)
        ALPHA  = ALPHA + 3.141593 / 12.0
 1820 CONTINUE
      ALPHA = 3.141593 / 2.0
      PLY(27) = 12.0
      DO 1830 I=14,26
        PLX(I) = PLX(27) + 1.00 * SIN(ALPHA)
        PLY(I) = PLY(27) + 0.25 * COS(ALPHA)
        ALPHA  = ALPHA + 3.141593 / 12.0
 1830 CONTINUE
      CALL GSFACI(GGRAYB)
      CALL GFA(26,PLX,PLY)
      PLX(27) = PLX(1)
      PLY(27) = PLY(1)

C***********************************************************************
C*****
C*****   Outline the disk storage icon with a white polyline.
C*****
C***********************************************************************

      CALL GSPLCI(GWHITE)
      CALL GPL(27,PLX,PLY)
      PLX(27) = 16.3
      PLY(27) = 13.0
      ALPHA = 3.141593 / 2.0
      DO 1840 I=1,13
        PLX(I) = PLX(27) + 1.00 * SIN(ALPHA)
        PLY(I) = PLY(27) + 0.25 * COS(ALPHA)
        ALPHA  = ALPHA + 3.141593 / 12.0
 1840 CONTINUE
      CALL GPL(13,PLX,PLY)

C***********************************************************************
C*****
C*****   Draw directional arrows between all the layers in the
C*****   flowchart.
C*****
C***********************************************************************

      CALL GSPLCI(GLBLUE)
      CALL GPL(10,PLATTX,PLATTY)
      CALL GPL(10,PAPX,PAPY)
      CALL GPL(10,GAGX,GAGY)
      CALL GPL(10,GPX,GPY)
      CALL GPL(10,PWX,PWY)
      DO 1850 I=1,10
        X(I) = PWX(I) + 3.0
 1850 CONTINUE
      CALL GPL(10,X,PWY)
      DO 1860 I=1,10
        X(I) = X(I) + 8.0
 1860 CONTINUE
      CALL GPL(10,X,PWY)

C***********************************************************************
C*****
C*****   Draw three light blue polymarkers inbetween the connected
C*****   workstations to denote that there are up to 'n'
C*****   workstations connected. Return the transformation view
C*****   back to view one.
C*****
C***********************************************************************

      CALL GSPMCI(GLBLUE)
      CALL GPM(3,WKMX,WKMY)
      CALL GSELNT(1)
      CALL RMENU(.TRUE.,CONT)
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: PRIMIT
C*****
C*****   Subroutine Function: This subroutine demonstrates all the
C*****                        output primitives and their
C*****                        attributes.
C*****
C*****   Calls Subroutines:   SETUPF,FINSHF,RCHOI,DEMOPM,DEMOPL
C*****                        DEMOTX,DEMOFA,DEMOCA,DEMOGD
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE PRIMIT
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,WTYPE,LCHSTR(7),LENGTH,CHNR,ERRIND,NUMELA
      INTEGER*4    I,J,PIXEL(9,9)

      REAL*4       FRAME(10),XMIN,XMAX,YMIN,YMAX,TXTX,TXTY
      REAL*4       PMX(13),PMY(13),ALPHA,PI,PX(5),PY(5)
      REAL*4       ZX(6),ZY(6),FA3(6),FA4(8),FA5(10)

      CHARACTER*12 MENU(7)
      CHARACTER*80 DTREC(7)

      COMMON       /WINF/ WKID,WTYPE
      COMMON       /TEXT/ TXTX,TXTY
      COMMON       /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA FRAME /0.000,0.000,0.995,0.995,0.000,
     *            0.000,0.995,0.995,0.000,0.000/
      DATA PIXEL /08,08,13,13,13,13,13,08,08,
     *            08,08,08,13,13,13,08,08,08,
     *            08,08,08,13,13,13,08,08,08,
     *            08,12,08,13,13,12,08,12,08,
     *            12,12,12,12,12,12,12,12,12,
     *            12,12,12,12,12,12,12,12,12,
     *            12,12,12,12,12,12,12,12,12,
     *            08,12,12,12,12,12,12,12,08,
     *            08,08,12,12,12,12,12,08,08/
      DATA FA3 /0.350,0.750,0.750,0.250,0.250,0.850/
      DATA FA4 /0.250,0.650,0.650,0.250,0.100,0.100,0.450,0.450/
      DATA FA5 /0.450,0.650,0.750,0.550,0.350,
     *          0.350,0.350,0.550,0.650,0.550/
      DATA MENU  /'POLYMARKER','POLYLINE','TEXT','FILL AREA',
     1             'CELL ARRAY','GDP','RETURN'/
      DATA LCHSTR /10,8,4,9,10,3,6/
      DATA NUMELA /7/

C***********************************************************************
C*****
C*****   Use pack data record to set up the prompt array for the
C*****   choice input device.
C*****
C***********************************************************************

      CALL GPREC(0,0,0,0.0,NUMELA,LCHSTR,MENU,NUMELA,
     *           ERRIND,LENGTH,DTREC)

C***********************************************************************
C*****
C*****   One big loop to keep drawing main output primitives menu
C*****   until the user selects the 'return' choice (lpfk 7).
C*****   Draw the outside border around the output primitives menu.
C*****
C***********************************************************************

 1900 CALL SETUPF(.TRUE.,.FALSE.)
      CALL GTXS(TXTX,0.886,6,'OUTPUT')
      CALL GTXS(TXTX,0.788,10,'PRIMITIVES')
      CALL FINSHF(.TRUE.)

C***********************************************************************
C*****
C*****   Draw six boxes which indicate the six choice alternatives:
C*****     Choice 1: Polymarker
C*****            2: Polyline
C*****            3: Text
C*****            4: Fill Area
C*****            5: Cell Array
C*****            6: GDPs
C*****
C***********************************************************************

      CALL GSCHH(0.060)
      CALL GSWN(2,0.0,1.0,0.0,1.0)
      CALL GSVP(2,0.075,0.325,0.395,0.695)
      CALL GSELNT(2)

C*************************************************************
C*****
C*****   Draw the polymarker choice box, place the text
C*****   'polymarker' in the box, and draw a circle of
C*****   polymarkers in the box.
C*****
C*************************************************************

      CALL GPL(5,FRAME(1),FRAME(6))
      CALL GSCHSP(0.0)
      CALL GTXS(0.100,0.900,12,'POLYMARKER  ')
      PMX(1) = 0.50
      PMY(1) = 0.45
      PI     = 3.141593
      ALPHA  = 0.0
      DO 1910 I=2,13
        PMX(I) = PMX(1) + 0.3 * SIN(ALPHA)
        PMY(I) = PMY(1) + 0.3 * COS(ALPHA)
        ALPHA  = ALPHA + 2.0 * PI / 12.0
 1910 CONTINUE
      CALL GSMK(GAST)
      CALL GSMKSC(1.0)
      CALL GSPMCI(GGREEN)
      CALL GPM(13,PMX,PMY)

C*************************************************************
C*****
C*****   Draw the polyline choice box, place the text
C*****   'polyline' in the box, and draw a star with
C*****   blue polylines in the box.
C*****
C*************************************************************

      CALL GSVP(2,0.375,0.625,0.395,0.695)
      CALL GPL(5,FRAME(1),FRAME(6))
      CALL GTXS(0.100,0.900,12,'POLYLINE    ')
      ALPHA  = 0.0
      DO 1920 I=1,5
        PX(I) = 0.5 +0.35 * SIN(ALPHA)
        PY(I) = 0.4 +0.35 * COS(ALPHA)
        ALPHA  = ALPHA + 2.0 * PI / 5.0
 1920 CONTINUE
      CALL GSLN(GSOLID)
      CALL GSLWSC(1.0)
      CALL GSPLCI(GLBLUE)
      DO 1940 I=1,2
        DO 1930 J=0,5
          K = MOD(I*J,5)
          ZX(J+1) = PX(K+1)
          ZY(J+1) = PY(K+1)
 1930   CONTINUE
        CALL GPL(6,ZX,ZY)
 1940 CONTINUE
      CALL GSPLCI(GYELOW)

C*************************************************************
C*****
C*****   Draw the text choice box, place the text 'text'
C*****   in the box, and draw 'GKS TEXT OUTPUT' in green,
C*****   red, and blue in the box.
C*****
C*************************************************************

      CALL GSVP(2,0.675,0.925,0.395,0.695)
      CALL GPL(5,FRAME(1),FRAME(6))
      CALL GTXS(0.100,0.900,12,'TEXT        ')
      CALL GSTXFP(1,GCHARP)
      CALL GSCHH(0.2)
      CALL GSTXCI(GGREEN)
      CALL GTXS(0.22,0.60,3,'GKS')
      CALL GSTXFP(2,GCHARP)
      CALL GSCHH(0.1)
      CALL GSTXCI(GRED)
      CALL GTXS(0.27,0.45,4,'TEXT')
      CALL GSTXFP(1,GSTRKP)
      CALL GSCHH(0.1)
      CALL GSTXCI(GLBLUE)
      CALL GSCHUP(1.0,1.0)
      CALL GTXS(0.18,0.33,1,'O')
      CALL GSCHUP(1.0,2.0)
      CALL GTXS(0.30,0.25,1,'U')
      CALL GSCHUP(1.0,4.0)
      CALL GTXS(0.42,0.20,1,'T')
      CALL GSCHUP(-1.0,4.0)
      CALL GTXS(0.57,0.19,1,'P')
      CALL GSCHUP(-1.0,2.0)
      CALL GTXS(0.68,0.23,1,'U')
      CALL GSCHUP(-1.0,1.0)
      CALL GTXS(0.79,0.30,1,'T')
      CALL GSCHUP(0.0,1.0)
      CALL GSTXFP(1,GSTRKP)
      CALL GSTXCI(GYELOW)
      CALL GSCHH(0.06)
      CALL GSCHSP(0.0)

C*************************************************************
C*****
C*****   Draw the fill area box, place the text 'fill
C*****   area' in the box, and draw some different solid
C*****   color shapes in the box.
C*****
C*************************************************************

      CALL GSVP(2,0.075,0.325,0.045,0.345)
      CALL GPL(5,FRAME(1),FRAME(6))
      CALL GTXS(0.100,0.900,12,'FILL AREA   ')
      CALL GSFASI(1)
      CALL GSFAIS(GSOLID)
      CALL GSFACI(GMAGNT)
      CALL GFA(3,FA3(1),FA3(4))
      CALL GSFACI(GMRED)
      CALL GFA(4,FA4(1),FA4(5))
      CALL GSFACI(GMGREN)
      CALL GFA(5,FA5(1),FA5(6))

C*************************************************************
C*****
C*****   Draw the cell array box, place the text 'cell
C*****   array' in the box, and draw a picture of a tree
C*****   using the cell array primitive.
C*****
C*************************************************************

      CALL GSVP(2,0.375,0.625,0.045,0.345)
      CALL GPL(5,FRAME(1),FRAME(6))
      CALL GTXS(0.100,0.900,12,'CELL ARRAY  ')
      CALL GCA(0.250,0.250,0.750,0.750,9,9,1,1,9,9,PIXEL)

C*************************************************************
C*****
C*****   Draw the GDPs box, place the text 'GDPs' in the
C*****   box, and display 'not available' in the box also
C*****   (as there is no GDP support in GKS-CO as yet).
C*****
C*************************************************************

      CALL GSVP(2,0.675,0.925,0.045,0.345)
      CALL GPL(5,FRAME(1),FRAME(6))
      CALL GTXS(0.100,0.900,12,'GDPs        ')
      CALL GTXS(0.05,0.4,13,'NOT AVAILABLE')

C*************************************************************
C*****
C*****   Initialize the choice device and call subroutine
C*****   RCHOI to wait for valid choice input. Analyze the
C*****   input and call the appropriate routine.
C*****
C*************************************************************

      CALL GSCHSP(0.0)
      CALL GSELNT(1)
      CALL GINCH(WKID,1,1,1,3,XMIN,XMAX,YMIN,YMAX,LENGTH,DTREC)
      CALL RCHOI(WKID,7,CHNR)
      IF (CHNR.EQ.1) THEN
        CALL DEMOPM
      ELSE
        IF (CHNR.EQ.2) THEN
          CALL DEMOPL
        ELSE
          IF (CHNR.EQ.3) THEN
            CALL DEMOTX
          ELSE
            IF (CHNR.EQ.4) THEN
              CALL DEMOFA
            ELSE
              IF (CHNR.EQ.5) THEN
                CALL DEMOCA
              ELSE
                IF (CHNR.EQ.6) THEN
                  CALL DEMOGD
                ELSE
                  GOTO 1950
                ENDIF
              ENDIF
            ENDIF
          ENDIF
        ENDIF
      ENDIF

C***********************************************************************
C*****
C*****   Continue the big loop to draw the output primitive menu.
C*****
C***********************************************************************

      GOTO 1900

C***********************************************************************
C*****
C*****   The user has setected the 'return' choice.
C*****
C***********************************************************************

 1950 RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: RCHOI
C*****
C*****   Subroutine Function: This subroutine will wait on valid
C*****                        input from the choice device (lpfks).
C*****                        If invalid input is received, put out
C*****                        an appropriate message and wait for
C*****                        another input choice.
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE RCHOI(WKID,MXCHNR,CHNR)

      INTEGER    WKID,MXCHNR,CHNR,STAT

 2000 CALL GRQCH(WKID,1,STAT,CHNR)
      IF (STAT.EQ.1) THEN
        IF ((CHNR.GT.0).AND.(CHNR.LE.MXCHNR)) THEN
          CALL GMSG(WKID,' ')
          GOTO 2010
        ELSE
          CALL GMSGS(WKID,22,' INVALID CHOICE NUMBER')
          STAT=0
        ENDIF
      ELSE
        CALL GMSGS(WKID,22,' CHOICE NOT SUCCESSFUL')
      ENDIF
      GOTO 2000
 2010 CONTINUE
      RETURN
      END

C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: RMENU
C*****
C*****   Subroutine Function: This subroutine will use pack data
C*****                        record to get the choice prompts
C*****                        set up and then it will initialize
C*****                        the choice device (lpfks) and call
C*****                        subroutine RCHOI to wait for valid
C*****                        choice input.
C*****
C*****   Calls Subroutines:   RCHOI
C*****
C*****   Input:   prompt - true: one prompt (return)
C*****                     false: two prompts (continue and return)
C*****
C*****   Output:  cont   - true:  user selected the continue choice
C*****                   - false: user selected the return choice
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE RMENU(PROMPT,CONT)
      INCLUDE 'gkspar.inc'

      INTEGER*4    WKID,LCHSTR(1),LCHST2(2),LENG1,ERRIND,CHNR
      REAL*4       XMIN,XMAX,YMIN,YMAX
      LOGICAL      PROMPT,CONT
      CHARACTER*8  MENU(1)
      CHARACTER*16 MENU2(2)
      CHARACTER*80 DTREC1(1)

      COMMON       /WINF/   WKID,WTYPE
      COMMON       /LIMITS/ XMIN,XMAX,YMIN,YMAX

      DATA         MENU   /'RETURN'/
      DATA         MENU2  /'CONTINUE','RETURN'/
      DATA         LCHSTR /6/
      DATA         LCHST2 /8,6/

      IF (PROMPT) THEN
        CALL GPREC(0,0,0,0.0,1,LCHSTR,MENU,LCHSTR,ERRIND,LENG1,DTREC1)
        NUMELA = 1
      ELSE
        CALL GPREC(0,0,0,0.0,2,LCHST2,MENU2,LCHST2,ERRIND,LENG1,DTREC1)
        NUMELA = 2
      ENDIF
      CALL GINCH(WKID,1,GOK,1,3,XMIN,XMAX,YMIN,YMAX,LENG1,DTREC1)
      CALL RCHOI(WKID,NUMELA,CHNR)
      IF ((CHNR.EQ.1).AND.(NUMELA.EQ.2)) THEN
        CONT = .TRUE.
      ELSE
        CONT = .FALSE.
      ENDIF
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: SETCOL
C*****
C*****   Subroutine Function: To set up the color table for the
C*****                        workstation indicated in WKID.
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE SETCOL

      INTEGER*4  WKID,I,WTYPE
      REAL*4     RED(15),GREEN(15),BLUE(15)
      COMMON     /WINF/ WKID,WTYPE

      DATA RED   /1.00,0.00,0.00,1.00,1.00,1.00,0.66,0.00,0.00,0.33,
     *            0.33,0.00,0.66,0.66,1.00/
      DATA GREEN /0.00,1.00,0.00,1.00,0.33,0.00,0.00,1.00,0.66,0.33,
     *            1.00,0.66,0.00,0.66,1.00/
      DATA BLUE  /0.00,0.00,1.00,0.00,0.00,1.00,0.66,1.00,1.00,0.66,
     *            0.33,0.00,0.00,0.66,1.00/

      DO 2100 I=1,15
         CALL GSCR(WKID,I,RED(I),GREEN(I),BLUE(I))
 2100 CONTINUE
      RETURN
      END



C***********************************************************************
C***********************************************************************
C*****
C*****   Subroutine Name: SETUPF
C*****
C*****   Subroutine Function: To draw the frame border for each
C*****                        screen of the demo.
C*****
C***********************************************************************
C***********************************************************************

      SUBROUTINE SETUPF(MFRAME,FRAME3)
      INCLUDE 'gkspar.inc'

      INTEGER*4  WKID,WTYPE
      REAL*4     PX1(5),PX2(5),PY1(5),PY2(5),PY4(5)
      REAL*4     TXTX,TXTY
      LOGICAL    MFRAME,FRAME3
      COMMON     /WINF/ WKID,WTYPE
      COMMON     /TEXT/ TXTX,TXTY

      DATA PX1 /0.000,0.000,1.000,1.000,0.000/
      DATA PX2 /0.015,0.015,0.985,0.985,0.015/
      DATA PY1 /0.000,1.000,1.000,0.000,0.000/
      DATA PY2 /0.765,0.985,0.985,0.765,0.765/
      DATA PY4 /0.915,0.985,0.985,0.915,0.915/

C***********************************************************************
C*****
C*****   Clear the workstation whether it is empty or not.
C*****
C***********************************************************************

      CALL GCLRWK(WKID,GALWAY)

C***********************************************************************
C*****
C*****   Draw the outside frame border.
C*****
C***********************************************************************

      CALL GCRSG (1259)
      CALL GSLN(GLSOLI)
      CALL GSLWSC(1.0)

C***********************************************************************
C*****
C*****   If drawing menu 3, the color indices will be invalid,
C*****   so a special index must be set up for the color wanted.
C*****   If not, set the color to yellow from the current color
C*****   table. Draw the top line.
C*****
C***********************************************************************

      IF (FRAME3) THEN
       CALL GSCR(WKID,15,1.0,1.0,0.0)
       CALL GSPLCI(15)
      ELSE
       CALL GSPLCI(GYELOW)
      ENDIF
      CALL GPL(5,PX1,PY1)

C***********************************************************************
C*****
C*****   Draw the frame that goes around the text at the top of
C*****   the mainframe.
C*****
C***********************************************************************

      TXTX=0.5
      IF (MFRAME) THEN
       CALL GPL(5,PX2,PY2)
       CALL GSCHH(0.072)
       TXTY=0.835
      ELSE
       CALL GPL(5,PX2,PY4)
       CALL GSCHH(0.036)
       TXTY=0.931
      ENDIF

C***********************************************************************
C*****
C*****   Set up the text attributes and go back to the routine
C*****   from which this subroutine was called.
C*****
C***********************************************************************

      CALL GSTXFP(GFONT2,GCHARP)
      CALL GSTXAL(GACENT,GAVNOR)
      CALL GSTXCI(GYELOW)
      CALL GSCHXP(1.0)
      RETURN
      end



