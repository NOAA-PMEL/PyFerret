	SUBROUTINE ROW_COORD_FMT( idim, cx, lo_ss, hi_ss, del_ss,
     .				  fmt, decimal, row_ww, cleft )

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
* left edge of each row of data.  The style will be (e.g.)
* "1.17N / 49:  26.31  26.06  26.01  26.04  26.14"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   7/9/90 - extracted from LIST_CMND_DATA
* V313:  31/10/94 - commas are mandatory in FORMAT statement on IBM/AIX
*		    changed format 3000 for all cases

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER idim, cx, lo_ss, hi_ss, del_ss, decimal, row_ww, cleft
	CHARACTER*(*) fmt

* local variable declarations:
	LOGICAL	GEOG_LABEL, date_str
	INTEGER DATE_PREC, FIELD_WIDTH,
     .		grid, j, places, row_ss, numbers
	REAL	BOX_SIZE, small
	REAL*8	TM_WORLD

* initialize
	row_ww = 1			! fallback ...
	IF ( cx .LE. 0 ) GOTO 500	! fallback format is "I / nnn:"
	grid = cx_grid( cx )

* how large is the smallest "delta" world coordinate to represent ?
	small = +1.E33
	DO 100 j = lo_ss, hi_ss, del_ss
	   small = MIN( small, BOX_SIZE( j, grid, idim ) )
 100	CONTINUE	      

* what precision is needed to represent this delta and world range ?
	date_str = idim .EQ. t_dim .AND. GEOG_LABEL(t_dim,grid) 
	IF ( date_str ) THEN
	   decimal = DATE_PREC( grid, small )
	ELSE
	   CALL GET_PREC_DIGITS(	cx_lo_ww(idim,cx),
     .					cx_hi_ww(idim,cx),
     .					small, places, decimal )
	ENDIF
	decimal = decimal + 1		! add 1 more, arbitrarily

* how many columns to fit the largest world coordinate ?
	row_ww = 0
	DO 200 j = lo_ss, hi_ss, del_ss
	   row_ww = MAX ( row_ww, FIELD_WIDTH
     .				  ( TM_WORLD(j,grid,idim,box_middle),
     .				    grid, idim, decimal, numbers ) )
 200	CONTINUE

* how many digits in the largest subscript ?
 500	row_ss = INT( LOG10(REAL(ABS(hi_ss))) ) + 1	! ABS for unspecified ss
	WRITE ( fmt, 3000 ) row_ss + 1	! blank before digits
! 3000	FORMAT ( '('' /''I',I1,''':'')' ) 	! pre-AIX
 3000	FORMAT ( '('' /'',I',I1,',',''':'')' )	! typical: (' /',I3,':') 

* total columns needed at left
	cleft = row_ww + 4 + row_ss		! "$ / ss:"

	RETURN
	END
