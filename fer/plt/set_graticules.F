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

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'command.parm'
	include 'plot_setup.parm'
	include 'xprog_state.cmn'
        include 'axis_inc.decl'
        include 'AXIS.INC'

* calling argument declarations:
	INTEGER	loch, locv, status

* internal variable declarations:
	LOGICAL linetype, dashtype, thick, colr, small

	INTEGER	TM_LENSTR1, STR_UPCASE, str_len, ii, i1, i2, i3, i4, 
     .      i5, i6, index1, index2, s1, s2, 
     .      colorcode, tic1, ntic, itic, iflag

        REAL thickness

	CHARACTER buff*1058
        CHARACTER msg*60

* internal parameter declarations
	INTEGER ncolors
	PARAMETER (ncolors = 6)

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
        IF (loch .GT. 0) THEN
           CALL EQUAL_STRING( cmnd_buff(qual_start(loch):qual_end(loch)),
     .                     buff, status )

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
           i4 = MIN(INDEX(buff(i3:i2), 'SMA'), INDEX(buff(i3:i2), ')' ) )
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
           i6 = MIN(INDEX(buff(i5:i2), 'LAR'), INDEX(buff(i5:i2), ')' ) )
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

           thickness = 1   ! default to thin line
           IF (thick) THEN
              s1 = INDEX(buff(index1:index2), 'THI') + index1 - 1
              s2 = INDEX(buff(s1:index2), ',') 
              IF (s2 .EQ. 0) THEN
                 s2 = index2
              ELSE
                 s2 = s2 + s1 - 2
              ENDIF 
              thickness = 2   ! default when /THICK specified
              IF (INDEX(buff(s1:s2), '=') .GT. 0) THEN
                 CALL EQUAL_VAL(buff(s1:s2), thickness, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
              ENDIF
              IF (thickness.LE.0 .OR. thickness.GT.3) GOTO 5150 
           ENDIF
 
           IF (colr) THEN
              s1 = INDEX(buff(index1:index2), 'COL') + index1 - 1
              s2 = INDEX(buff(s1:index2), ',')
              IF (s2 .EQ. 0) THEN
                 s2 = index2
              ELSE
                 s2 = s2 + s1 - 2
              ENDIF 
              hcolor(itic) = 1   ! default to black
              IF (INDEX(buff(s1:s2), '=') .GT. 0) THEN
                 CALL EQUAL_COLOR( buff(s1:s2), colorcode, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
                 hcolor(itic) = colorcode
              ELSE
                 WRITE (msg, 40) buff(s1:s2)
  40             FORMAT ( 'No color code given for /HGRAT=',A , 
     .                    '. Using black')
                 CALL WARN (msg)
              ENDIF

              IF (hcolor(itic).LE.0 .OR. hcolor(itic) .GT. 19) GOTO 5160
           ENDIF

* if colorcode = 19 then line is thin, white.
           IF (hcolor(itic) .LT. 19) 
     .        hcolor(itic) = ncolors*(thickness-1) + hcolor(itic) ! set color and thick

           index1 = i5
           index2 = i6

        ENDDO  ! back to to small htics, if requested.

* 
 4999   CONTINUE

*  /VGRAT

        IF (locv .GT. 0) THEN
           CALL EQUAL_STRING( cmnd_buff(qual_start(locv):qual_end(locv)),
     .                     buff, status )

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

           i4 = MIN(INDEX(buff(i3:i2), 'SMA'), INDEX(buff(i3:i2), ')' ) )
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

           i6 = MIN(INDEX(buff(i5:i2), 'LAR'), INDEX(buff(i5:i2), ')' ) )
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

           thickness = 1   ! default to thin line
           IF (thick) THEN
              s1 = INDEX(buff(index1:index2), 'THI') + index1 - 1
              s2 = INDEX(buff(s1:index2), ',') 
              IF (s2 .EQ. 0) THEN
                 s2 = index2
              ELSE
                 s2 = s2 + s1 - 2
              ENDIF 
              thickness = 2   ! default when /THICK specified
              IF (INDEX(buff(s1:s2), '=') .GT. 0) THEN
                 CALL EQUAL_VAL(buff(s1:s2), thickness, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
              ENDIF
              IF (thickness.LE.0 .OR. thickness.GT.3) GOTO 5150 
           ENDIF
 
           IF (colr) THEN
              s1 = INDEX(buff(index1:index2), 'COL') + index1 - 1
              s2 = INDEX(buff(s1:index2), ',')
              IF (s2 .EQ. 0) THEN
                 s2 = index2
              ELSE
                 s2 = s2 + s1 - 2
              ENDIF 
              vcolor(itic) = 1   ! default to black
              IF (INDEX(buff(s1:s2), '=') .GT. 0) THEN
                  CALL EQUAL_COLOR( buff(s1:s2), colorcode, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
                 vcolor(itic) = colorcode
              ELSE
                 WRITE (msg, 60) buff(s1:s2)
  60             FORMAT ( 'No color code given for /VGRAT=',A , 
     .                    '. Using black')
                 CALL WARN (msg)
              ENDIF
              IF (vcolor(itic).LE.0 .OR. vcolor(itic) .GT. 19) GOTO 5160
           ENDIF

* if colorcode = 19 then line is thin, white.
           IF (vcolor(itic) .LT. 19) 
     .        vcolor(itic) = ncolors*(thickness-1) + vcolor(itic) ! set color and thick

           index1 = i5
           index2 = i6

        ENDDO  ! back to to small htics, if requested.

* success
        RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status,
     .		buff(:str_len)//' midmatched parentheses', *5000 )
 5050	CALL ERRMSG( ferr_syntax, status,
     .		buff(:str_len), *5000 )
 5150	CALL ERRMSG( ferr_invalid_command, status,
     .		'"'//buff(:str_len)//
     .          '": thickness must be between 1 and 3',
     .		*5000 )
 5160	CALL ERRMSG( ferr_invalid_command, status,
     .		'"'//buff(s1:s2)//'": must be between 1 and 3',
     .		*5000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'zero vector length on /LENGTH=', *5000 )
	END
