	SUBROUTINE PPL_DASH_RESTORE

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
* restore the PLOT+ dash lenghts after they have been changed by PLOT/DASH=

* V541: 3/25  *acm*

	! variables DN1,DN2,UP1,UP2
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
        include 'ppl_dash_save.cmn'

* calling argument declarations

* local variable declarations:
	INTEGER line_number

* For dash lines set up by /DASH=  reset to previous settings which
* may be from a PPL LINE command and so should persist.

      IF (linenum_save .GT. 0) THEN
         DO 100 line_number = 1, linenum_save

            dash_changed(line_number) = .FALSE.
            dn1(line_number) = dashsave(line_number,1)
            up1(line_number) = dashsave(line_number,2)
            dn2(line_number) = dashsave(line_number,3)
            up2(line_number) = dashsave(line_number,4) 
  100    CONTINUE

         dashsave(line_number,1) = 0.
         dashsave(line_number,2) = 0.
         dashsave(line_number,3) = 0.
         dashsave(line_number,4) = 0.

      linenum_save = 0
      ENDIF

      RETURN
      END
