	DOUBLE PRECISION FUNCTION TM_WORLD ( isubscript, grid_num, idim,
     .					     where_in_box )

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
* convert a subscript value on a grid into a "world" coordinate
* position within the grid box corresponding to the given subscript is
* determined by the value where_in_box

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 04/16/86
* revision 1.00 - 03/10/87 - line data in COMMON/*VARIABLES*/
* revision 1.10 - 03/23/87 - added new TMAP library: time axes, file var. names
* revision 1.20 - 02/18/88 - incorporated "regular" axes; trapped "unspecified"
*			     subscripts
* revision 1.30 - 08/09/88 - mod to reflect storage of box coords not sizes (mv)
* revision 1.31 - 11/29/88 - munknown, mnormal changed to mpsnorm, mpsunkn (mv)
* revision 2.00 - 12/12/88 - changed to reflect storage of box locations instead
*			     of box sizes for irregular axis... (mv)
* revision 2.10 - 01/25/89 - Added ability to handle regular modulo axis
* revision 2.11 - 01/07/91 - added "*" to comment above for RISC compiler
* 10/16/95 - for Ferret V4.20 - added recursive response for child axes of
*			irregularly-spaced axes
* *kob* 10/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed them

* Include files
#ifdef unix
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* calling argument declarations:
	INTEGER	isubscript, grid_num, idim, where_in_box

* internal variable declarations:
	LOGICAL	recursive
	INTEGER	iaxis, isub, line_len, rmod
	REAL*8	TM_WORLD_RECUR, midpoint, box_size, tempwld

* initialize
	iaxis    = grid_line( idim, grid_num )
	IF ( iaxis .EQ. mpsnorm
     .	.OR. iaxis .EQ. mpsunkn
     .	.OR. isubscript .EQ. unspecified_int4 ) THEN
	  TM_WORLD = unspecified_val8
	  RETURN
	ENDIF
	line_len = line_dim( iaxis )

* if this is a child (e.g. every Nth point) of an irregularly-spaced
* axis then use a recursive C routine to find the answer
	recursive = iaxis .GT. max_lines
	IF ( recursive ) THEN
	  recursive = line_parent(iaxis) .NE. 0 
	  IF ( recursive ) THEN
	    TM_WORLD = TM_WORLD_RECUR(
     .			isubscript,iaxis,where_in_box,max_lines,
     .			line_mem,line_parent,line_class,line_dim,line_start,
     .			line_delta,line_subsc1,line_modulo,line_regular )
	    RETURN
	  ENDIF
	ENDIF

* force given subsc to data range as appropriate for modulo or non-modulo axes
	IF ( line_modulo( iaxis ) ) THEN
	  isub = MOD(isubscript-1,line_len) + 1
	  IF (isub .LE. 0) isub = isub + line_len
	ELSE
	  isub = MIN( line_len, MAX( 1, isubscript ) )
	ENDIF
* regularly spaced points
	IF ( line_regular( iaxis ) ) THEN
* ... calculate midpoint and box_size values
	  midpoint = line_start(iaxis) + (isub-1)*line_delta(iaxis)
	  box_size = line_delta(iaxis)
* ... now where within the grid box ?
	  IF ( where_in_box .EQ. box_lo_lim ) THEN
	    tempwld = midpoint - ( box_size / 2. )
	  ELSEIF ( where_in_box .EQ. box_middle ) THEN
	    tempwld = midpoint
	  ELSE
	    tempwld = midpoint + ( box_size / 2. )
	  ENDIF
	  IF ( line_modulo( iaxis ) ) THEN
	    IF (isubscript .LE. 0) THEN
	      rmod = line_len * ( isubscript/line_len - 1 )
	    ELSE
	      rmod = line_len * ( (isubscript-1)/line_len )
	    ENDIF
	    TM_WORLD = tempwld + rmod*line_delta(iaxis)
	  ELSE
	    TM_WORLD = tempwld
	  ENDIF
	ELSE
* not regularly spaced points
* ... keep subscript within bounds ( between 1 and dimension of axis )
	  isub  = isub + line_subsc1(iaxis) - 1
* ... now where within the grid box ?
	  IF ( where_in_box .EQ. box_lo_lim ) THEN
	    tempwld = line_mem(isub+line_len)
	  ELSEIF ( where_in_box .EQ. box_middle ) THEN
	    tempwld = line_mem(isub)
	  ELSE
	    tempwld = line_mem(isub+line_len+1)
	  ENDIF
	  IF ( line_modulo( iaxis ) ) THEN
	    IF (isubscript .LE. 0) THEN
	      rmod = isubscript/line_len - 1
	    ELSE
	      rmod = (isubscript-1)/line_len
	    ENDIF
	    TM_WORLD = tempwld + rmod*
     .	                     ( line_mem(line_subsc1(iaxis)+2*line_len) -
     .	                       line_mem(line_subsc1(iaxis)+  line_len) )
	  ELSE
	    TM_WORLD = tempwld
	  ENDIF
	ENDIF

	RETURN
	END


