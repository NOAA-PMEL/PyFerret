

	SUBROUTINE LINE_FACTS( lun, idim, grid, orient, uvar )

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
* print basic facts about an axis defined in memory

* New 12/15:
* last argument: uvar is positive if we are to describe the 
* grid subset for the user-variable given.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/1/88 - extracted from SHOW_GRID - axis limits --> box_middle
* revision 0.1 - 8/16/88 - elimiinated units, expanded title field
* revision 0.2 - 2/6/89 - added modulo axis indicator 
* V200:  9/22/89 - added "orient" for axes without particular orientations
* SUN port 3/92 *sh* - "T" format specifiers cannot move to left
*                    - changed format 3020
* V230:  6/22/92 - allow for long axis names by truncating the title field
*        1/93 - corrected left-over bug in format 3020 from SUN port A15->A14
* V320: 12/20/94 - Put "(-)" along with "m" and "r" so it doesn't get
*		truncated now that DEPTH(units) is the label
*       12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V510 *sh* 12/99 - show units even if line has no orientation
* V68  *acm 12/11 - if npts > 1.e7 the format overflowed. Up the size.
* V697  12/15 *acm* Changes to add details on SHOW GRID uvar to indicate 
*                   compressed or subset of axes

	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
#	include "tmap_dset.parm"
	external xgt_grid_data
	include	'ferret.parm'
	include 'xtext_info.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
	include 'xvariables.cmn'

* calling argument declarations
	INTEGER	  lun, idim, grid, orient, uvar

* local variable declarations:
	LOGICAL	BKWD_AXIS, GEOG_LABEL, calendar, is_uvar_2lim,
     .          is_uvar_1lim, is_uvar_nolim
	INTEGER TM_LENSTR1, TM_SUBSC, lo, hi, i, line, dlen, ndec, status,
     .          nlen, naxlen, int1, slen, i1ss, inss
	CHARACTER AX_TITLE*32, TM_FMT*20, lo_str*20, hi_str*20, direction*1,
     .		  title*32, reg*1, modflag*1, truncated*1, frmt*80,
     .		  lo_uvar*20, hi_uvar*20, uvar_desc*40, cstr*3, countstr*20
	REAL*8	TM_WORLD, lo_val8, hi_val8, pts

* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 5 )

* pull line from grid table
	line = grid_line( idim, grid )

* line ok ?
	risc_buff = ' '
	IF ( line .EQ. munknown ) THEN
	   title = 'unknown'
	   WRITE ( risc_buff, 3010 ) title, ww_dim_name( idim )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ELSEIF ( line .EQ. mnormal ) THEN
	   title = 'normal'
	   WRITE ( risc_buff, 3010 ) title, ww_dim_name( idim )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ELSEIF ( line_name(line) .EQ. char_init)  THEN
	   title = 'normal'
	   WRITE ( risc_buff, 3010 ) title, ww_dim_name( idim )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ELSE

* get nicely formatted limits (end box outer edges)
	   calendar = idim .EQ. t_dim .AND. GEOG_LABEL( idim, grid )	
	   lo_val8 = TM_WORLD( 1, grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 5 		! accurate to minutes
	   ELSE
	      ndec = -sig_figs	! 5 sig places w/out extra zeros
	   ENDIF
	   CALL TRANSLATE_TO_WORLD( lo_val8, idim, grid, ndec, lo_str )
	   hi_val8 = TM_WORLD( line_dim(line), grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 5 
	   ELSE
	      ndec = -sig_figs
	   ENDIF
	   CALL TRANSLATE_TO_WORLD( hi_val8, idim, grid, ndec, hi_str )

* axis oriented backwards ?
	   IF ( BKWD_AXIS(idim, grid) ) THEN
	      direction = '-'
	   ELSE
	      direction = ' '
	   ENDIF

* axis title
	   IF ( orient .EQ. 0 ) THEN
	      title = line_units(line)
	      IF (title .EQ. ' ') THEN
	         dlen = 1
	      ELSE
	         dlen = TM_LENSTR1(title)
	         title = '('//line_units(line)(:dlen)//')'
	         dlen = dlen + 2
	      ENDIF
	   ELSE
	      title = AX_TITLE( idim, grid, dlen )
	   ENDIF
!	   title = title(:dlen)//direction	! 12/94 removed

* regularly spaced ?
	   IF ( line_regular( line ) ) THEN
	     reg = 'r'
	   ELSE
	     reg = 'i'
	   ENDIF

* modulo ?
	   IF ( line_modulo( line ) ) THEN
	     modflag = 'm'
	   ELSE
	     modflag = ' '
	   ENDIF

* allow for long axis names (6/92)
           nlen = MAX( TM_LENSTR1(line_name(line)), 9 )

	   naxlen = INT(0.5+DLOG10(DFLOAT(line_dim(line))) ) 

* report information about the subsetting or compressing of the grid.

* From merge_grid:
* There are subtlities associated with limits that are given in the UVAR
* definition.  These are classified in uvar_given under these codings:
* ("gvn" indicates lo/hi limits were given in square brackets)
* ("gnl","lo","lohi" and "xact" encode special requirements of transformations)
*
*    -1   uvlim_irrelevant- limits irrelevant: NEW=10.1
*    -2   uvlim_hidden    - limits hidden inside on another axis: NEW=FFT(v)
*     0   uvlim_gvn_gnl   - limits given in a gen'l way: NEW=U[K=1:5]
*     1   uvlim_gvn_lo    - limits that require exact lo match: NEW=U[K=1:5@IIN]
*     2   uvlim_gvn_lohi  - limits require exact lohi match: NEW=U[K=1:5@WEQ]
*     3   uvlim_gvn_xact  - limits that reduce to a pt.: NEW=U[K=1:5@AVE]
*     4   uvlim_needed    - no limit info. given
*     5   uvlim_need_lo   - no lims but transf. needs xact lo:NEW=U[K@IIN]
*     6   uvlim_need_lohi - no lims but transf. needs xact lohi:NEW=U[K=@WEQ]
*     7   uvlim_need_xact - no lims but transf. needs xact: NEW=U[K@AVE]
*   100   uvlim_unknown

* these are more elaborate than necessary - would allow for reporting the subset range
	   is_uvar_2lim = .FALSE.
	   is_uvar_1lim = .FALSE.
	   is_uvar_nolim = .FALSE.

	   uvar_desc = ' '

	   IF (uvar .GT. 0) THEN 

* Subsets on axis, compressing transform, ...


* Subsets orange in transform or expression
	   IF ( uvar_given(idim,uvar) .EQ. uvlim_gvn_gnl  .OR. 
     .          uvar_given(idim,uvar) .EQ. uvlim_gvn_lo   .OR. 
     .          uvar_given(idim,uvar) .EQ. uvlim_gvn_lohi )  THEN
	      uvar_desc = ' (subset)'
	      is_uvar_2lim = .TRUE.
	      IF (uvar_lo_ss(uvar, idim) .NE. unspecified_int4) THEN
	         i1ss = uvar_lo_ss(uvar, idim)
	         inss = uvar_hi_ss(uvar, idim)
	         is_uvar_1lim = ( i1ss .EQ. inss )
	      ELSE
	         i1ss =  TM_SUBSC ( uvar_lo_ww(idim, uvar), grid, idim, -1 )
	         inss =  TM_SUBSC ( uvar_hi_ww(idim, uvar), grid, idim, -1 )
	         is_uvar_1lim = ( i1ss .EQ. inss )
	      ENDIF

	      IF (is_uvar_1lim) THEN
	         is_uvar_2lim = .FALSE.
	         uvar_desc = ' (subset 1 pt)'
	      ELSE
	         countstr = TM_FMT(FLOAT(inss-i1ss), 15, 20, slen)
	         uvar_desc = ' (subset '//countstr(:slen)//' pts)'
	      ENDIF
	   ENDIF

* Compressing
*   Single-point subsets are marked as gvn_xact (for conformability?)
*   which would result in "compressed" here. 
*   So if gvn_xact and hi=lo, call that a single-point subset.
*   This means that var[x=5@max] is also reported as point subset. 

	   IF ( uvar_given(idim,uvar) .EQ. uvlim_gvn_xact)  THEN
	      uvar_desc = ' (compressed)'
	      is_uvar_2lim = .TRUE.
	      IF (uvar_lo_ss(uvar, idim) .NE. unspecified_int4) THEN
	         is_uvar_1lim = ( uvar_lo_ss(uvar, idim) .EQ. uvar_hi_ss(uvar, idim) )
	      ELSE
	         is_uvar_1lim = ( uvar_lo_ww(idim, uvar) .EQ. uvar_hi_ww(idim, uvar) )
	      ENDIF
	      IF (is_uvar_1lim) THEN
	         uvar_desc = ' (subset 1 pt)'
		 is_uvar_2lim = .FALSE.
	      ENDIF
	   ENDIF

c Compressing transfm, w/o limits

	   IF ( uvar_given(idim,uvar) .EQ. uvlim_need_xact) THEN
	      is_uvar_nolim = .TRUE.
	      uvar_desc = ' (compressed)'
	   ENDIF

* Grid-changing function: uvar_given = uvar_hidden for idim

	   IF (uvar_given(idim,uvar) .EQ. uvlim_hidden )  THEN
	      is_uvar_2lim = .TRUE.
	      uvar_desc = ' (axis from grid-changing function)'
	   ENDIF

c	   IF (uvar_given(idim,uvar) .EQ. uvlim_need_lo  .OR. 
c     .         uvar_given(idim,uvar) .EQ. uvlim_need_lohi )  THEN 
c	      is_uvar_nolim = .TRUE.
c	      uvar_desc = ' (expression, limits not set)'
c	   ENDIF


	   ENDIF !  uvar, reporting subset, compression

* write info
	   int1 = 34 - (naxlen+1)
	   naxlen = naxlen + 1
	   IF (naxlen .LT. 10) THEN
	      WRITE (frmt, 3020) int1, naxlen
	   ELSE
	      WRITE (frmt, 3030) int1, naxlen
	   ENDIF

           WRITE ( risc_buff, frmt )	line_name ( line )(:nlen),
     .				title (:41-nlen),
     .				line_dim  ( line ),
     .				modflag,
     .				reg,
     .				direction,
     .				lo_str,
     .				hi_str

* If there is a subset or compression, add that information

	   IF (is_uvar_2lim .OR. is_uvar_1lim .OR. is_uvar_nolim) THEN

	      WRITE ( risc_buff, frmt )	line_name ( line )(:nlen),
     .				title (:41-nlen),
     .				line_dim  ( line ),
     .				modflag,
     .				reg,
     .				direction,
     .				lo_str,
     .				hi_str,
     .				uvar_desc
	   ENDIF

	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	ENDIF
	RETURN

 3010	FORMAT ( T2,A8,T12,A1 )

 3020   FORMAT ('(T2,A,1X,A,T',I2,',I',I1',3A1,T39,A20,T60,A20,T80,A)')
 3030   FORMAT ('(T2,A,1X,A,T',I2,',I',I2',3A1,T39,A20,T60,A20,T80,A)')

	END
