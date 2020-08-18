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
* V697  12/15 *acm* Changes to formatting the output that shows subsets
* V698 2/16 *acm*  For ticket 1786: if mode_upcase is canceled write axis name
*                  using original upper/lowercase spelling
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V72  3/17 *acm* Fix ticket 2518: The number of points reported in a subset was incorrect
* v74  *acm*  1/18 Fixing issue 1575, setting uvar_given for result of grid-changing functions
*                  that return results on abstract axes.
* v75  *acm* 3/18  Issue 1856, changes to allow for timesteps in fractional seconds
* v74  *acm* 3/18  Backing off the fix for 1575, see comments 3/16/2018
* V751 6/19 *acm* TM_FMT rewrite: formatting of lon/lat coordinate strings
* V751 6/19 *acm* For DSG data indicate the obs dimension

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
#	include "tmap_dset.parm"
	include 'implicit.parm'
	include 'xdset_info.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'xtext_info.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
        include 'calendar.decl'
        include 'calendar.cmn'

* calling argument declarations
	INTEGER	  lun, idim, grid, orient, uvar

* local variable declarations:
	LOGICAL	BKWD_AXIS, GEOG_LABEL, ITSA_TRUEMONTH_AXIS, TM_FPEQ,
     .          calendar, original, true_month, ss_ok, its_dsg_ragged, 
     .          its_dsg, its_cmpnd

	INTEGER TM_LENSTR1, TM_SUBSC, TM_GET_CALENDAR_ID, STR_SAME,
     .          lo, hi, i, line, dlen, ndec, status,nlen, naxlen, 
     .          int1, slen, i1ss, inss, cal_id, dsg_orient, obsdimlen, 
     .          feature_line
	CHARACTER AX_TITLE*32, TM_FMT*48, lo_str*28, hi_str*28, direction*1,
     .		  title*32, reg*1, modflag*1, truncated*1, frmt*80,
     .		  lo_uvar*20, hi_uvar*20, uvar_desc*40, cstr*3, countstr*20,
     .		  axname*64, buff*20
	REAL*8	TM_WORLD, lo_val8, hi_val8, pts

* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 5 )

* pull line from grid table
	line = grid_line( idim, grid )

* is this a DSG ragged contiguous "axis"
	its_dsg_ragged = idim.NE.e_dim
     .		   .AND. line_dsg_dset(line).NE.pline_not_dsg

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
	      ndec = sig_figs	! 5 sig places w/out extra zeros
	      ndec = -5  ! 
	      IF (i .EQ. y_dim) ndec = -3
	   ENDIF
	   true_month = ITSA_TRUEMONTH_AXIS (line)
	   IF (true_month) ndec = 2
	   CALL TRANSLATE_TO_WORLD( lo_val8, idim, grid, ndec, lo_str )
	   hi_val8 = TM_WORLD( line_dim(line), grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 5 
	   ELSE
	      ndec = sig_figs
	      ndec = -5  ! 
	      IF (i .EQ. y_dim) ndec = -3
	   ENDIF
	   IF (true_month) ndec = 2

	   CALL TRANSLATE_TO_WORLD( hi_val8, idim, grid, ndec, hi_str )

	   IF (.NOT.TM_FPEQ(lo_val8, hi_val8) .AND. STR_SAME (lo_str, hi_str) .EQ. 0) THEN
	      ndec=9
	      CALL TRANSLATE_TO_WORLD( lo_val8, idim, grid, ndec, lo_str )
	      CALL TRANSLATE_TO_WORLD( hi_val8, idim, grid, ndec, hi_str )
	   ENDIF

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
	   IF (true_month) THEN
               cal_id = TM_GET_CALENDAR_ID (line_cal_name (line))
	       IF (cal_id .NE. d360) reg = 'i'
	   ENDIF

* modulo ?
	   IF ( line_modulo( line ) ) THEN
	     modflag = 'm'
	   ELSE
	     modflag = ' '
	   ENDIF

* allow for long axis names (6/92)
	   original = .FALSE.
	   CALL CHOOSE_LINE_NAME(line, original, axname)
           nlen = MAX( TM_LENSTR1(axname), 9 )

	   naxlen = INT(0.5+DLOG10(DFLOAT(line_dim(line))) ) 

* report information about the subsetting or compressing of the grid.

* FERRET DOCUMENTATION: implicit grid modifications of LET vars (e.g. @AVE)
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

* Subsets on axis, compressing transform, ... All this is
* skipped if it's a SHOW AXIS or SHOW GRID gridname

	   IF (uvar .GT. 0) THEN 
	      uvar_desc = 'full'

* Subsets orange in transform or expression
	   IF ( uvar_given(idim,uvar) .EQ. uvlim_gvn_gnl  .OR. 
     .          uvar_given(idim,uvar) .EQ. uvlim_gvn_lo   .OR. 
     .          uvar_given(idim,uvar) .EQ. uvlim_gvn_lohi )  THEN
cc	      ss_ok = .TRUE.
	      IF (uvar_lo_ss(uvar, idim) .NE. unspecified_int4) THEN
	         i1ss = uvar_lo_ss(uvar, idim)
	         inss = uvar_hi_ss(uvar, idim)
	      ELSE
cc		 ss_ok = (uvar_lo_ww(idim, uvar).NE.unspecified_val8 .AND.  
cc     .                    uvar_hi_ww(idim, uvar).NE.unspecified_val8)
cc	         IF (ss_ok) THEN
		    i1ss =  TM_SUBSC ( uvar_lo_ww(idim, uvar), grid, idim, -1 )
	            inss =  TM_SUBSC ( uvar_hi_ww(idim, uvar), grid, idim, -1 )
cc	         ENDIF
	      ENDIF

! bit of a hack; see the fix for issue 1575 in abstract-axis in is_do_gc_fcn.F
cc	      IF (ss_ok) THEN  
	       IF (i1ss .EQ. inss) THEN
	          uvar_desc = '1 pt'
	       ELSE
	          countstr = TM_FMT(FLOAT(inss-i1ss+1), 15, 20, slen)
	          uvar_desc = countstr(:slen)//' pts'
	       ENDIF
cc	      ENDIF  ! ss_ok
	   ENDIF

* Compressing
*   Single-point subsets are marked as gvn_xact (for conformability?)
*   which would result in "compressed" here. 
*   So if gvn_xact and hi=lo, call that a single-point subset.
*   This means that var[x=5@max] is also reported as point subset. 

	   IF ( uvar_given(idim,uvar) .EQ. uvlim_gvn_xact)  THEN
	      IF (uvar_lo_ss(uvar, idim) .NE. unspecified_int4) THEN
	         i1ss = uvar_lo_ss(uvar, idim)
	         inss = uvar_hi_ss(uvar, idim)
	      ELSE
	         i1ss =  TM_SUBSC ( uvar_lo_ww(idim, uvar), grid, idim, -1 )
	         inss =  TM_SUBSC ( uvar_hi_ww(idim, uvar), grid, idim, -1 )
	      ENDIF

	      IF (i1ss .EQ. inss) THEN
	         uvar_desc = '1 pt'
	      ELSE
	         uvar_desc = 'compressed'
	      ENDIF

	   ENDIF

c Compressing transfm, w/o limits

	   IF ( uvar_given(idim,uvar) .EQ. uvlim_need_xact) THEN
	      uvar_desc = 'compressed'
	   ENDIF

* Grid-changing function: uvar_given = uvar_hidden for idim

	   IF (uvar_given(idim,uvar) .EQ. uvlim_hidden )  THEN
	      uvar_desc = '(axis from grid-changing func)'
	   ENDIF

	   ENDIF !  uvar, reporting subset, compression

* Special description to flag the DSG feature orientation axis
* If the grid is pointing to th dsg_xlate_grid don't show the 
* axes other than the one in the dsg-orientation direction.

	   IF (its_dsg_ragged) THEN
	      CALL TM_DSG_FACTS( grid, dsg_orient, obsdimlen, 
     .                  feature_line, its_dsg, its_cmpnd )
     
	      IF (its_cmpnd) dsg_orient = z_dim
	      IF (its_dsg_ragged .AND. idim.LE.t_dim .AND. idim.NE.dsg_orient) THEN
	         title = 'normal'
	         WRITE ( risc_buff, 3010 ) title, ww_dim_name( idim )
		 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
		 GOTO 5000
	      ENDIF
	      IF (idim .LE. t_dim) uvar_desc = " " 
	      IF (idim .EQ. dsg_orient) uvar_desc = "Variable lengths"
	   ENDIF 

* create format adapted to name length (??)
	   int1 = 34 - (naxlen+1)
	   naxlen = naxlen + 1
	   IF (naxlen .LT. 10) THEN
	      WRITE (frmt, 3020) int1, naxlen
	   ELSE
	      WRITE (frmt, 3030) int1, naxlen
	   ENDIF

* write info (see below -- this may get over-written)
* ?? *sh* 4/17 - could we clean this up? just append the uvar_desc?
	   WRITE ( risc_buff, frmt )	axname(:nlen),
     .				title (:41-nlen),
     .				line_dim  ( line ),
     .				modflag,
     .				reg,
     .				direction,
     .				lo_str,
     .				hi_str

* If there is a subset or compression, add that information. This is
* skipped if it's a SHOW AXIS or SHOW GRID gridname
	   IF (uvar .GT. 0) THEN
	      WRITE ( risc_buff, frmt ) axname (:nlen),
     .				title (:41-nlen),
     .				line_dim  ( line ),
     .				modflag,
     .				reg,
     .				direction,
     .				lo_str,
     .				hi_str,
     .				uvar_desc
	   ENDIF
	   IF (true_month) THEN
	      dlen = TM_LENSTR1(risc_buff)
	      IF (dlen.GT.1) risc_buff = risc_buff(:dlen)//' (True Monthly)'
	   ENDIF
	   

	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	ENDIF

 5000	RETURN

 3010	FORMAT ( T2,A8,T12,A1 )

 3020   FORMAT ('(T2,A,1X,A,T',I2,',I',I1',3A1,T39,A20,T60,A20,T80,A)')
 3030   FORMAT ('(T2,A,1X,A,T',I2,',I',I2',3A1,T39,A20,T60,A20,T80,A)')

	END
