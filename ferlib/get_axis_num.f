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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
*  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.



*
*
      SUBROUTINE GET_AXIS_NUM(axisnum, axisstart, axisend,
     .                        axisname, axisunit, axiscoords,
     .                        numcoords, axistype, errmsg, lenerrmsg)
      IMPLICIT NONE

      INCLUDE 'tmap_dims.parm'
      INCLUDE 'xtm_grid.cmn_text'
      INCLUDE 'ferret_lib.parm'

*     Passed arguments
      CHARACTER*(*) axisname, axisunit, errmsg
      INTEGER       axisnum, axisstart, axisend,
     .              numcoords, axistype, lenerrmsg
      REAL*8        axiscoords
      DIMENSION     axiscoords(numcoords)

*     Function declarations
      INTEGER TM_LENSTR, TM_UNIT_ID, STR_UPCASE
      LOGICAL TM_LEGAL_NAME

*     Local arguments
      CHARACTER*(64) up_axisname
      INTEGER axisunitcode, i, j, k, sts
      REAL*8  delta, eps, value
      LOGICAL regular
      LOGICAL modu
      REAL*8  modulen

*     Sanity check
      IF ( numcoords .LT. 1 ) THEN
          errmsg = 'Non-positive numcoords passed to GET_AXIS_NUM'
          lenerrmsg = TM_LENSTR(errmsg)
          RETURN
      ENDIF

*     Uppercase the name for comparisons
      IF ( axisname .NE. '' ) THEN
          sts = STR_UPCASE(up_axisname, axisname)
          IF ( .NOT. TM_LEGAL_NAME(up_axisname) ) THEN
              errmsg = 'Invalid axis name given'
              lenerrmsg = TM_LENSTR(errmsg)
              RETURN
          ENDIF
      ELSE
          up_axisname = ''
      ENDIF

*     Get the code number of known units; returns zero for unknown units
      axisunitcode = TM_UNIT_ID(axisunit)

      delta = (axiscoords(numcoords) - axiscoords(1)) / (numcoords - 1)
      eps = abs(delta) * 1.0E-7
      if ( eps .LT. 1.0E-32 )
     .    eps = 1.0E-32
*     Check for regular coordinates
      regular = .TRUE.
      DO 10 k = 2, numcoords-1
         value = axiscoords(1) + (k-1) * delta
         IF ( abs(value - axiscoords(k)) .GT. eps ) THEN
             regular = .FALSE.
             GOTO 20
         ENDIF
   10 CONTINUE

*     Crude assumption:
*        longitude are modular with a modulo of 360 deg
*        and nothing else is modular.
   20 IF ( axistype .EQ. AXISTYPE_LONGITUDE ) THEN
          modu = .TRUE.
          modulen = 360.0
      ELSE
          modu = .FALSE.
          modulen = 0.0
      ENDIF

*     See if a line already exists with the provided info
      IF ( up_axisname .NE. '' ) THEN
          DO 100 k = 1, max_dyn_lines
*             Check quick and easy things before strings
              IF ( .NOT. regular .EQV. line_regular(k) )
     .            GOTO 100
              IF ( .NOT. modu .EQV. line_modulo(k) )
     .            GOTO 100
              IF ( modu .AND. (modulen .NE. line_modulo_len(k)) )
     .            GOTO 100
              IF ( axisunitcode .NE. line_unit_code(k) )
     .            GOTO 100
              IF ( (axisunitcode .EQ. 0) .AND.
     .             (axisunit .NE. line_units(k)) )
     .            GOTO 100
*             ??? Do we really care if the name matches
*                 as long as everything else does ???
              IF ( up_axisname .NE. line_name(k) )
     .            GOTO 100
*             Check the coordinates - may be a subset
              IF ( regular ) THEN
                  IF ( abs(delta - line_delta(k)) .GT. eps )
     .                GOTO 100
                  DO 30 j = 1, line_dim(k)
                      value = line_start(k) + (j-1) * line_delta(k)
                      IF ( abs(value - axiscoords(1)) .LT. eps ) THEN
                          i = j + numcoords - 1
                          IF ( i .GT. line_dim(k) )
     .                        GOTO 100
*                         Successfully found the regular axis
                          axisnum = k
                          axisstart = j
                          axisend = i
                          GOTO 500
                      ENDIF
  30              CONTINUE
              ELSE
*                 Find the first coordinate
                  DO 40 j = 1, line_dim(k)
                      value = line_mem(line_subsc1(k) + j - 1)
                      IF ( abs(value - axiscoords(1)) .LT. eps )
     .                    GOTO 50
   40             CONTINUE
                  GOTO 100
*                 Check the rest of the coordinates
   50             IF ( (j + numcoords - 1) .GT. line_dim(k) )
     .                GOTO 100
                  DO 60 i = 2, numcoords
                      value = line_mem(line_subsc1(k) + j + i - 2)
                      IF ( abs(value - axiscoords(i)) .GE. eps )
     .                    GOTO 100
   60             CONTINUE
*                 Successfully found the irregular axis
                  axisnum = k
                  axisstart = j
                  axisend = i
                  GOTO 500
              ENDIF
  100    CONTINUE
      ENDIF

*     If we got here, there is no line matching the description.
*     So create one.
*     Stubbed
      axisnum = -1
      axisstart = 0
      axisend = 0

  500 lenerrmsg = 0
      WRITE(*,*) 'axisnum = ', axisnum
      WRITE(*,*) 'axisstart = ', axisstart
      WRITE(*,*) 'axisend = ', axisend
      RETURN

      END



*
*
      SUBROUTINE GET_TIME_AXIS_NUM(axisnum, axisstart, axisend,
     .                             axisname, caltype, axiscoords,
     .                             numcoords, errmsg, lenerrmsg)
      IMPLICIT NONE

      INCLUDE 'ferret_lib.parm'

*     Passed arguments
      CHARACTER*(*) axisname, errmsg
      INTEGER       axisnum, axisstart, axisend, caltype,
     .              axiscoords, numcoords, lenerrmsg
      DIMENSION     axiscoords(6,numcoords)

*     Local arguments
      INTEGER k

      WRITE(*,*) 'axisname = ', axisname
      WRITE(*,*) 'caltype = ', caltype
      WRITE(*,*) 'numcoords = ', numcoords
      IF ( numcoords .GE. 1 ) THEN
          WRITE(*,*) 'axiscoords(*,1) = ', (axiscoords(k,1),k=1,6)
          WRITE(*,*) 'axiscoords(*,', numcoords, ') = ',
     .                (axiscoords(k,numcoords),k=1,6)
      ENDIF

*     Stubbed
      axisnum = -1
      axisstart = 1
      axisend = 1

      lenerrmsg = 0
      RETURN

      END

