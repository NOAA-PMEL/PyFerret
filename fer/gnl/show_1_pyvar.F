      SUBROUTINE SHOW_1_PYVAR(lun, ivar, full)

*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administrations (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* output some lines describing a pyvar (cat_pystat_var) variable
*
      IMPLICIT NONE

      include 'ferret.parm'
      include 'tmap_dims.parm'
      include 'xprog_state.cmn'
      include 'xpyvar_info.cmn'
      include 'xrisc.cmn'
      include 'xtext_info.cmn'
      include 'xtm_grid.cmn_text'

* arguments
      INTEGER lun, ivar
      LOGICAL full

* function return values
      INTEGER      TM_LENSTR1
      CHARACTER*48 LOHI_STRING
      CHARACTER*48 TM_FMT
      CHARACTER*6  LEFINT
      REAL*8       TM_WORLD

* local variables
      CHARACTER*256 myline
      CHARACTER*17  axstr(nferdims), fltstr
      INTEGER       j, k, q, starti(nferdims), grid, listdims
      REAL*8        lo, hi

      listdims = nferdims
      IF ( .NOT. mode_6d_lab ) listdims = 4

* if the code name is long, print it on a separate line
      myline = pyvar_code(ivar)
      IF ( TM_LENSTR1(myline) .GT. 8 ) THEN
          risc_buff = ' '
          WRITE (risc_buff, '(1X,A)') pyvar_code(ivar)
          CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
          myline = ' '
      ENDIF

* grid axis limits
      DO 120 k = 1,listdims
          IF ( grid_line(k,pyvar_grid_number(ivar)) .EQ. mnormal) THEN
              axstr(k) = '       ...'
              starti(k) = 6
          ELSE
              WRITE(axstr(k),'(I8)') pyvar_grid_start(k,ivar)
*             just use the last three characters unless more is needed
              starti(k) = 6
              DO 100 j = 1,5
                  IF ( axstr(k)(j:j) .NE. ' ' ) THEN
                      starti(k) = j
                      GOTO 110
                  ENDIF
 100          CONTINUE
 110          CONTINUE
              axstr(k)(9:9) = ':'
              axstr(k)(10:) = LEFINT(pyvar_grid_end(k,ivar),j)
          ENDIF
 120  CONTINUE

* write the basic information line
      IF ( listdims .EQ. nferdims ) THEN
          WRITE (risc_buff, 200) myline, pyvar_title(ivar),
     .                           (axstr(k)(starti(k):15),k=1,nferdims)
 200      FORMAT(1X,A8,1X,A31,T42,6(A))
      ELSE
          WRITE (risc_buff, 205) myline, pyvar_title(ivar),
     .                           (axstr(k)(starti(k):15),k=1,4)
 205      FORMAT(1X,A8,1X,A31,T42,4(A))
      ENDIF
      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 

* write more information if full output requested
      IF ( full ) THEN
          grid = pyvar_grid_number(ivar)
*         "<units> on grid <name> with <value> for missing data"
          j = TM_LENSTR1(pyvar_units(ivar))
          k = TM_LENSTR1(grid_name(grid))
          fltstr = TM_FMT(pyvar_missing_flag(ivar),7,14,q)
          myline = '             ' // pyvar_units(ivar)(:j) //
     .           ' on grid ' // grid_name(grid)(:k) // ' with ' //
     .           fltstr(:q) // ' for missing data'
          CALL SPLIT_LIST(pttmode_explct, lun, myline, 0)
* ... world coordinate limits
          myline = ' '
          j = 1
          DO 300 k = 1,3
              IF ( grid_line(k,grid) .NE. mnormal ) THEN
                  lo = TM_WORLD(pyvar_grid_start(k,ivar), 
     .                          grid, k, box_lo_lim)
                  hi = TM_WORLD(pyvar_grid_end(k,ivar), 
     .                          grid, k, box_hi_lim )
                  risc_buff(:256) = myline(:j)
                  myline = risc_buff(:j) // ww_dim_name(k) // '=' //
     .                   LOHI_STRING(lo,hi,k,grid,':',ax_dec_pt(k),q)
     .            // '  '
                  j = MIN(64,j+4+q)
               ENDIF
 300      CONTINUE
          CALL SPLIT_LIST(pttmode_explct, lun,
     .                    '            ' // myline(:j), j+12)
      ENDIF

      RETURN
      END


