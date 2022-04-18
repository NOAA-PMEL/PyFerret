        SUBROUTINE SET_GRATICULES(loch, locv, status )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* parse qualifier text to set up horizontal and vertical graticule lines 
* extending the tic marks across plots.
*
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* Ferret v5.54   2/04
* V570: *acm* 5/04 - add MODE GRATICULE[:argument] 
* V570 *acm* 6/30/04 VGRAT with no argument needs to default to lines
* v580 *acm* 11/04   PLOT/HGRAT gave both horiz and vertical graticules
*                    need to initialize buff.
* V6.2  *acm& 2/09 Fix choice of color settings with /PEN and /THICK when we have 
*                  increased the num of colors with MODE linecolors
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* PyFr  *kms* 7/13 EQUAL_COLOR now returns a color ID, and EQUAL_THICK 
*                  returns a pen ID from a color ID and a thickness.
*                  Only standard lines accepted for graticules at this time.
*       *acm* 3/22 Pyferret issue 51 Non-standard graticule lines, either 
*                  custom color settings or non-integer thickness is not 
*                  implemented. Make this an error not a warning.

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'ferret.parm'
        include 'errmsg.parm'
        include 'command.parm'
        include 'plot_setup.parm'
        include 'xprog_state.cmn'
        include 'axis_inc.decl'
        include 'AXIS.INC'

* calling argument declarations:
        INTEGER loch, locv, status

* internal variable declarations:
        LOGICAL linetype, dashtype, thick, colr, small

        INTEGER TM_LENSTR1, STR_UPCASE, str_len, ii, i1, i2, i3, i4, 
     .      i5, i6, index1, index2, s1, s2, s3,
     .      colorcode, tic1, ntic, itic, iflag, ncolors, pencode

        REAL*4 thickness
        REAL val

        CHARACTER buff*1058
        CHARACTER msg*60

! Default, standard tics, no lines across plot.

        buff = ' '

        DO 100 ii = 1, 4
          hgrat(ii,1) = 0.
          hgrat(ii,2) = 0.
          vgrat(ii,1) = 0.
          vgrat(ii,2) = 0.
100     CONTINUE

        hcolor(1) = 1   ! default color is black
        hcolor(2) = 1
        vcolor(1) = 1
        vcolor(2) = 1

* If MODE GRAT was set, make those settings.  Any settings
* made on the command line will override these.

        IF (grat_on .AND. ((loch .EQ. 0) .AND. (locv .EQ. 0)) ) THEN
           DO 101 ii = 1, 4
             hgrat(ii,1) = -1.
             hgrat(ii,2) = 0.
             vgrat(ii,1) = -1.
             vgrat(ii,2) = 0.
101        CONTINUE

           iflag = STR_UPCASE (buff, mode_grat_buff)
        ENDIF

* HGRAT
        IF (.NOT. grat_on) buff = ' ' 
        IF (loch .GT. 0) THEN
           CALL EQUAL_STRING(cmnd_buff(qual_start(loch):qual_end(loch)),
     .                       buff, status )

* /HGRAT with no argument defaults to lines at large tic marks.

           DO 105 ii = 1, 4
             hgrat(ii,1) = -1.
105        CONTINUE

        ENDIF

        IF ( buff .EQ. ' ' ) GOTO 4999

        str_len = TM_LENSTR1(buff)
        i1 = 1
        IF (buff(1:1) .EQ. '(') i1 = 2
        i2 = str_len

        tic1 = 1   ! LARGE tics by default
        ntic = 1
        
        index1 = i1
        index2 = i2

        IF ( INDEX(buff(i1:i2), 'LAR') .GT. 0) THEN
           tic1 = 1   ! LARGE tics
           i3 = INDEX(buff(i1:i2), 'LAR' )
           i4 = MIN(INDEX(buff(i3:i2), 'SMA'), INDEX(buff(i3:i2), ')' ))
           IF (i4 .GT. 0) THEN
              i4 = i4 + i3 - 2
           ELSE
              I4 = i2
           ENDIF

           index1 = i3
           index2 = i4
        ENDIF

        small = .FALSE.
        IF ( INDEX(buff(i1:i2), 'SMA') .GT. 0) small = .TRUE.
        IF (small) THEN
           i5 = INDEX(buff(i1:i2), 'SMA')
           i6 = MIN(INDEX(buff(i5:i2), 'LAR'), INDEX(buff(i5:i2), ')' ))
           IF (i6 .GT. 0) THEN
              i6 = i6 + i5 - 2
           ELSE
              i6 = i2
           ENDIF

           ntic = 2

* If only small requsted, reset lines at large tics to off

           IF (INDEX(buff(i1:i2), 'LAR') .LE. 0) THEN
              tic1 = 2

              DO 106 ii = 1, 4
                hgrat(ii,1) = -0.
106           CONTINUE

              index1 = i5
              index2 = i6

           ENDIF
        ENDIF

        DO itic = tic1, ntic

* By default use solid lines
           DO 110 ii = 1, 4
              hgrat(ii,itic) = -1.
110        CONTINUE

           IF (buff(index1:index1) .EQ. '(') index1 = index1 + 1
           IF (buff(index2:index2) .EQ. ')') index2 = index2 - 1

           linetype = INDEX(buff(index1:index2), 'LIN') .GT. 0
           dashtype = INDEX(buff(index1:index2), 'DAS') .GT. 0
           thick = INDEX(buff(index1:index2), 'THI') .GT. 0
           colr = INDEX(buff(index1:index2), 'COL') .GT. 0

           IF (linetype .AND. dashtype) THEN
              CALL WARN (
     .         '/HGRAT specified with both LINE and DASH, using LINE')
              dashtype = .FALSE.
           ENDIF

           IF (linetype) THEN
              hgrat(1,itic) = -1.  ! for solid lines
              hgrat(2,itic) = -1.
              hgrat(3,itic) = -1.
              hgrat(4,itic) = -1.

           ELSE IF (dashtype) THEN
              hgrat(1,itic) = 0.03  ! set default values
              hgrat(2,itic) = 0.05
              hgrat(3,itic) = 0.0
              hgrat(4,itic) = 0.0
           ENDIF

           hcolor(itic) = 1
           thickness = 1.0   ! default to a thin black line 

           IF (colr) THEN
              s1 = INDEX(buff(index1:index2), 'COL') + index1 - 1
*             First assume a comma ends the color specification
              s2 = INDEX(buff(s1:index2), ',')
              IF (s2 .EQ. 0) THEN
                 s2 = index2
              ELSE
*                Do not include the comma
                 s2 = s2 + s1 - 2
              ENDIF 
*             But if there is a open parenthesis before the comma, 
*             a closed parenthesis ends the color specification
              s3 = INDEX(buff(s1:s2),'(')
              IF ( s3 .GT. 0 ) THEN
                 s2 = INDEX(buff(s1:index2), ')')
                 IF (s2 .EQ. 0) THEN
                    GOTO 5100
                 ELSE
*                   Include the closing parenthesis
                    s2 = s2 + s1 - 1
                 ENDIF
              ENDIF
              IF (INDEX(buff(s1:s2), '=') .GT. 0) THEN
                 CALL EQUAL_COLOR(buff(s1:s2), colorcode, thickness, status)
                 IF ( status .NE. ferr_ok ) GOTO 5000
                 hcolor(itic) = colorcode
              ELSE
                 WRITE (msg, 40) buff(s1:s2)
  40             FORMAT ( 'No color code given for /HGRAT=', A , 
     .                    '. Using black')
                 CALL WARN (msg)
              ENDIF
           ENDIF

           IF (thick) THEN
              s1 = INDEX(buff(index1:index2), 'THI') + index1 - 1
              s2 = INDEX(buff(s1:index2), ',') 
              IF (s2 .EQ. 0) THEN
                 s2 = index2
              ELSE
                 s2 = s2 + s1 - 2
              ENDIF

*    In the graticule setting, a value of THICK that's not 1,2, or 3,
*    or /THICK with no argument, is not implemented.
*    Check this using EQUAL_VAL; a call to equal_thick may change an 
*    existingsetting on a line-plot command

	      CALL EQUAL_VAL( buff(s1:s2), val, status )
	      IF (val .EQ. unspecified_val4) THEN
	         ! ok
	      ELSE
	         IF (val .NE. INT(VAL) .OR. val.LT.1 .OR. val.GT.3) GOTO 5200
	      ENDIF

*             get the pen for the color and the thickness specified in buff
              CALL EQUAL_THICK(buff(s1:s2), thickness, hcolor(itic), 
     .                         pencode, status)
              IF ( status .NE. ferr_ok ) GOTO 5000
           ELSE
*             get the pen for the color and the default thickness
              CALL EQUAL_THICK(' ', thickness, hcolor(itic), 
     .                         pencode, status)
              IF ( status .NE. ferr_ok ) GOTO 5000
           ENDIF
 
*          At this time, only standard lines are accepted
           ncolors = mode_arg(pmode_linecolors,1)
           IF ( (pencode .LT. 1) .OR. (pencode .GT. 3*ncolors) ) GOTO 5200

*          Assign the pen to use
           hcolor(itic) = pencode

           index1 = i5
           index2 = i6

        ENDDO  ! back to to small htics, if requested.

* 
 4999   CONTINUE

*  /VGRAT

        IF (.NOT. grat_on) buff = ' ' 
        IF (locv .GT. 0) THEN
           CALL EQUAL_STRING(cmnd_buff(qual_start(locv):qual_end(locv)),
     .                     buff, status )

* /VGRAT with no argument defaults to lines at large tic marks.

           DO 115 ii = 1, 4
             vgrat(ii,1) = -1.
115        CONTINUE

        ENDIF
        IF ( buff .EQ. ' ' ) GOTO 5000  ! done

        str_len = TM_LENSTR1(buff)

        i1 = 1
        IF (buff(1:1) .EQ. '(') i1 = 2
        i2 = str_len

        tic1 = 1   ! LARGE tics by default
        ntic = 1
        
        index1 = i1
        index2 = i2

        IF ( INDEX(buff(i1:i2), 'LAR') .GT. 0) THEN
           tic1 = 1   ! LARGE tics
           i3 = INDEX(buff(i1:i2), 'LAR' )

           i4 = MIN(INDEX(buff(i3:i2), 'SMA'), INDEX(buff(i3:i2), ')' ))
           IF (i4 .GT. 0) THEN
              i4 = i4 + i3 - 2
           ELSE
              I4 = i2
           ENDIF
           
           index1 = i3
           index2 = i4
        ENDIF

        small = .FALSE.
        IF ( INDEX(buff(i1:i2), 'SMA') .GT. 0) small = .TRUE.
        IF (small) THEN
           i5 = INDEX(buff(i1:i2), 'SMA')

           i6 = MIN(INDEX(buff(i5:i2), 'LAR'), INDEX(buff(i5:i2), ')' ))
           IF (i6 .GT. 0) THEN
              i6 = i6 + i5 - 2
           ELSE
              i6 = i2
           ENDIF
           
           ntic = 2

* If only small requsted, reset lines at large tics to off

           IF (INDEX(buff(i1:i2), 'LAR') .LE. 0) THEN
              tic1 = 2

              DO 116 ii = 1, 4
                vgrat(ii,1) = -0.
116           CONTINUE

              index1 = i5
              index2 = i6

           ENDIF
        ENDIF

        DO itic = tic1, ntic

* By default use solid lines
           DO 120 ii = 1, 4
              vgrat(ii,itic) = -1.
120        CONTINUE

           IF (buff(index1:index1) .EQ. '(') index1 = index1 + 1
           IF (buff(index2:index2) .EQ. ')') index2 = index2 - 1

           linetype = INDEX(buff(index1:index2), 'LIN') .GT. 0
           dashtype = INDEX(buff(index1:index2), 'DAS') .GT. 0
           thick = INDEX(buff(index1:index2), 'THI') .GT. 0
           colr = INDEX(buff(index1:index2), 'COL') .GT. 0

           IF (linetype .AND. dashtype) THEN
              CALL WARN (
     .         '/VGRAT specified with both LINE and DASH, using LINE')
              dashtype = .FALSE.
           ENDIF

           IF (linetype) THEN
              vgrat(1,itic) = -1.  ! for solid lines
              vgrat(2,itic) = -1.
              vgrat(3,itic) = -1.
              vgrat(4,itic) = -1.

           ELSE IF (dashtype) THEN
              vgrat(1,itic) = 0.03 ! set default values
              vgrat(2,itic) = 0.05
              vgrat(3,itic) = 0.0
              vgrat(4,itic) = 0.0
           ENDIF

           vcolor(itic) = 1
           thickness = 1.0   ! default to thin black line

           IF (colr) THEN
              s1 = INDEX(buff(index1:index2), 'COL') + index1 - 1
*             First assume a comma ends the color specification
              s2 = INDEX(buff(s1:index2), ',')
              IF (s2 .EQ. 0) THEN
                 s2 = index2
              ELSE
*                Do not include the comma
                 s2 = s2 + s1 - 2
              ENDIF 
*             But if there is a open parenthesis before the comma, 
*             a closed parenthesis ends the color specification
              s3 = INDEX(buff(s1:s2),'(')
              IF ( s3 .GT. 0 ) THEN
                 s2 = INDEX(buff(s1:index2), ')')
                 IF (s2 .EQ. 0) THEN
                    GOTO 5100
                 ELSE
*                   Include the closing parenthesis
                    s2 = s2 + s1 - 1
                 ENDIF
              ENDIF
              IF (INDEX(buff(s1:s2), '=') .GT. 0) THEN
                 CALL EQUAL_COLOR(buff(s1:s2), colorcode, thickness, status)
                 IF ( status .NE. ferr_ok ) GOTO 5000
                 vcolor(itic) = colorcode
              ELSE
                 WRITE (msg, 60) buff(s1:s2)
  60             FORMAT ( 'No color code given for /VGRAT=', A , 
     .                    '. Using black')
                 CALL WARN (msg)
              ENDIF
           ENDIF

           IF (thick) THEN
              s1 = INDEX(buff(index1:index2), 'THI') + index1 - 1
              s2 = INDEX(buff(s1:index2), ',') 
              IF (s2 .EQ. 0) THEN
                 s2 = index2
              ELSE
                 s2 = s2 + s1 - 2
              ENDIF 

*    In the graticule setting, a value of THICK that's not 1,2, or 3,
*    or /THICK with no argument, is not implemented.
*    Check this using EQUAL_VAL; a call to equal_thick may change an 
*    existingsetting on a line-plot command

	      CALL EQUAL_VAL( buff(s1:s2), val, status )
	      IF (val .EQ. unspecified_val4) THEN
	         ! ok
	      ELSE
	         IF (val .NE. INT(VAL) .OR. val.LT.1 .OR. val.GT.3) GOTO 5200
	      ENDIF

*             get the pen for the color and the thickness specified in buff
              CALL EQUAL_THICK(buff(s1:s2), thickness, vcolor(itic), 
     .                         pencode, status)
              IF ( status .NE. ferr_ok ) GOTO 5000
           ELSE
*             get the pen for the color and the default thickness
              CALL EQUAL_THICK(' ', thickness, vcolor(itic), 
     .                         pencode, status)
              IF ( status .NE. ferr_ok ) GOTO 5000
           ENDIF
 
*          At this time, only standard lines are accepted
           ncolors = mode_arg(pmode_linecolors,1)
           IF ( (pencode .LT. 1) .OR. (pencode .GT. 3*ncolors) ) GOTO 5200

*          Assign the pen to use
           vcolor(itic) = pencode

           index1 = i5
           index2 = i6

        ENDDO  ! back to to small htics, if requested.

* success
        RETURN

* error exit
 5000        RETURN
 5100        CALL ERRMSG( ferr_syntax, status,
     .                buff(:str_len)//' mismatched parentheses', *5000 )
 5200        CALL ERRMSG(ferr_syntax, status,
     .                'For graticules only standard line colors ' //
     .                'and thicknesses are accepted.', *5000 )

        END
