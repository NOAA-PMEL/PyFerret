	SUBROUTINE TM_Write_Gridfile ( gnum, lunit )
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
*
* Writes a commented gridfile on lunit for the grid specified by gnum
*
* Written 03/89 by M. Verschell for PMEL/TMAP
*
* revision 0.00 - 03/21/89 - initial incarnation with 1 grid only
* rev 0.1 6/92 *sh* - added EXTERNAL declaration needed on port to MAC
* *kob* 10/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed them
* V50 - 2/99 *kob* modifications for y2k preparedness.  call 
*                  GET_DATE_AND_TIME which uses FDATE to get
*                  date and time rather than DATE and TIME
* arguments:
	INTEGER*4	gnum, lunit

#ifdef unix
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xio.cmn_text'
	external xio_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XIO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif
* 2/99 kob - need only one var to store date and time w
	CHARACTER*8	ctime
	CHARACTER*9	cdate
        EXTERNAL        TRIM    ! added 6/92
	CHARACTER*20	TRIM
	CHARACTER*132	stars, line1, line2, time1, grid1, crds1, buff
	INTEGER*4	icnt, lnum, istat, npts, iloc, ibox, jcnt, pos

	stars =	'********************************************'//
     .		'********************************************'//
     .		'********************************************'
	line1 = '* Line definitions first: name, number of po'//
     .		'ints, ordering, and defined by start,delta o'//
     .		'r coordinate values (given below)          *'
	line2 = '* "ordering" may be "WE" (longitude), "SN"(l'//
     .		'atitude), "UD"(depth), "TI"(time) or "NA"(no'//
     .		'n-standard values)                         *'
	time1 = '* Next time line definitions: timeaxis linen'//
     .		'ame, followed by t0                         '//
     .		'                                            '
	grid1 = '* Next comes grid definitions: gridname and '//
     .		'rotation, followed by lines they contain (al'//
     .		'ong with inner/outer product flag)          '
	crds1 = '* Lastly, coordinate/box size pairs for thos'//
     .		'e lines not specified by start,delta: line n'//
     .		'ame followed by format for coordinates      '

*
* Check to see if grid asked for has data
	IF (grid_name(gnum) .EQ. char_init16) THEN
	  WRITE (tooplu,'(A,I5)') 'No grid defined at position',gnum
	  GOTO 9990
	ENDIF
*
* Write out DATE/TIME info
*	CALL DATE (cdate)
*	CALL TIME (ctime)
* 2/99 *kob* modifications for y2k preparedness.  call GET_DATE_AND_TIME
*            to get date and time
	CALL GET_DATE_AND_TIME(cdate,ctime)
	WRITE (lunit,'(4A)') '* GRID FILE CREATED ',cdate,':',ctime
*
* Write out line info header
	WRITE (lunit,'(A)') stars
	WRITE (lunit,'(A)') line1
	WRITE (lunit,'(A)') line2
	WRITE (lunit,'(A)') stars
*
* Write out line info
	DO 100 icnt = 1,4
	  lnum = grid_line(icnt, gnum)
	  IF (lnum .EQ. mpsnorm .OR. lnum .EQ. mpsunkn) GOTO 100
	  buff = ' '
	  WRITE (buff(1:55),1000)
     .				'LINE ',
     .				TRIM(line_name(lnum),20),
     .				line_dim(lnum),
     .				TRIM(line_direction(lnum),10),
     .				TRIM(line_units(lnum),10)
 1000	  FORMAT(A5,A20,I10,2A10)
	  IF (line_regular(lnum)) THEN
	    WRITE (buff(56:100),1100)	'START,DELTA',
     .					line_start(lnum),
     .					line_delta(lnum)
 1100	    FORMAT(A15,2(1PE15.6))
	  ELSE
	    WRITE (buff(56:100),1200)	'GIVEN BELOW',
     .					'XXXX             '
 1200	    FORMAT(A15,A30)
	  ENDIF
	  IF (line_modulo(lnum)) WRITE (buff(101:110),1300) 'MODULO'
 1300	  FORMAT(A10)
	  WRITE (lunit,'(A)') buff
  100	CONTINUE
*
* Write out time info header
	WRITE (lunit,'(A)') stars
	WRITE (lunit,'(A)') time1
	WRITE (lunit,'(A)') stars
*
* Write out time info (IF ANY)
	lnum = grid_line(4, gnum)
	IF (lnum .EQ. mpsunkn) GOTO 110
	WRITE (lunit,1400) 'TIME ',line_name(lnum),line_t0(lnum)
 1400	FORMAT(A5,A20,A25)
*
* Write out grid info header
  110	WRITE (lunit,'(A)') stars
	WRITE (lunit,'(A)') grid1
	WRITE (lunit,'(A)') stars
*
* Write out grid info
	buff = ' '
	WRITE (buff(1:35),1500)
     .				'GRID ',
     .				TRIM(grid_name(gnum),20),
     .				grid_rotation(gnum)
 1500	FORMAT(A5,A20,F10.4)
	DO icnt = 1,4
	  lnum = grid_line(icnt, gnum)
	  pos = 24*icnt
	  IF (lnum .EQ. mpsnorm) THEN
	    WRITE (buff(pos+12:pos+29),1600) 'NORMAL'
 1600	    FORMAT(A18)
	  ELSE IF (lnum .EQ. mpsunkn) THEN
	    WRITE (buff(pos+12:pos+29),1600) 'UNKNOWN'
	  ELSE
	    WRITE (buff(pos+12:pos+29),1600) TRIM(line_name(lnum),18)
	  ENDIF
	  IF (grid_out_prod(icnt,gnum)) THEN
	    WRITE (buff(pos+30:pos+35),1700) 'OUTER'
 1700	    FORMAT(A6)
	  ELSE
	    WRITE (buff(pos+30:pos+35),1700) 'INNER'
	  ENDIF
	ENDDO
	WRITE (lunit,'(A)') buff
*
* Write out crds info header
	WRITE (lunit,'(A)') stars
	WRITE (lunit,'(A)') crds1
	WRITE (lunit,'(A)') stars
*
* Write out crds info (IF ANY)
	DO 120 icnt = 1,4
	  lnum = grid_line(icnt, gnum)
	  IF (lnum .EQ. mpsnorm .OR. lnum .EQ. mpsunkn) GOTO 120
	  IF (.NOT. line_regular(lnum)) THEN
	    npts = line_dim(lnum)
	    iloc = line_subsc1(lnum)
	    ibox = iloc + npts
	    buff = ' '
	    WRITE (buff(1:60),1800)
     .				'COORDS    ',
     .				TRIM(line_name(lnum),20),
     .				'(8(1PE15.6))',
     .				line_mem(ibox)
 1800	    FORMAT(A10,A20,A15,1PE15.6)
	    WRITE (lunit,'(A)') buff
	    WRITE (lunit,1900)	(line_mem(jcnt),
     .				line_mem(jcnt+npts+1)-
     .	                        line_mem(jcnt+npts),   jcnt=iloc,ibox-1)
 1900	    FORMAT(8(1PE15.6))
	  ENDIF
  120	ENDDO
*
* We done
 9990	RETURN
	END
*
********
*
	CHARACTER*(*) FUNCTION TRIM ( in_str, pos )
*
* Trims in_str of blanks and tabs and right justifies the output to pos
*
	CHARACTER*(*)	in_str
	INTEGER*4	pos, i, j
*
	TRIM = ' '
	DO i = 1,LEN(in_str)
	  IF (in_str(i:i) .NE. ' ') THEN
	    DO j = LEN(in_str),i,-1
	      IF (in_str(j:j) .NE. ' ') THEN
	        TRIM(pos+i-j:pos) = in_str(i:j)
	        RETURN
	      ENDIF
	    ENDDO
	  ENDIF
	ENDDO
	END
