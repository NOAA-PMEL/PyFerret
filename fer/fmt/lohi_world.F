	CHARACTER*(*) FUNCTION LOHI_WORLD( idim,cx,to_word,full_prec,slen )

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
* create a string describing the world coordinate boundaries along an axis
* for example: 160E:140W or 10m (if lo depth = hi)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/26/87
* revision 0.1 - 2/25/88 - changed TRANSLATE_TO_WORLD call to allow non-lat/long
*			   formatting.
* revision 0.2 - 3/25/88 - number of decimals set in SET MODE command
*			   and possibly reverse ordering of time limits since
*			   stored times may be tsteps (for a particular T axis)
*			   or -1*(secs_from_BC)  (independent of partic. axis)
* revision 0.3 - 8/12/88 - negative times not converted to time steps
* revision 1.0 -  1/10/89- split off actual string creation into LOHI_STRING
* V200:  5/19/89 - 4D symmetrical - point limits have ww coords at box edges 
*	 3/22/90 - for cx.LE.cx_last use last successful grid
*	 9/11/90 - corrected bug: IF test treated cx_buff as a stored region
* V300:   2/3/93 - the 9/11/90 bug fix prevents the default region for being
*                  formatted
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V690 *sh* 1/14 - added kludge to display limits from aux var regrid by avg
* V698  *acm* 5/16 Ticket 2405: MODE CALENDAR:auto
* V698 *sh* 4/16 hack to format E axis limits as integers
*	         Note:  this is a poor fix.  See 4/16 note below
* V74+ *acm* 8/18 DSG dataset? If so find coordinates on the dsg_xlate_grid 
* V761 8/20 *acm* Longer LOHI_WORLD, LOHI_STRING strings higher-precision outputs

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'	! added 3/90
	include 'xfr_grid.cmn'		! added 3/90
	include 'xunits.cmn_text'
	include 'xdset_info.cmn_text'


* calling argument declarations
	LOGICAL	full_prec
	INTEGER	idim, cx, slen
	CHARACTER*(*) to_word

* internal variable declarations:
	LOGICAL	SCALAR_GRID_LINE, TM_ITSA_DSG_RAGGED, TM_ITSA_DSG, 
     .          ITSA_DSG_POINTS, 
     .          aux_span_compress, its_float, its_dsg
	INTEGER	DATE_PREC, TM_DSG_DSET_FROM_GRID, 
     .          dset, sub_len, ndec, grid, line, status, 
     .          cx_try, cat, var, loss, hiss, delss, j, decimal
	REAL*8	TM_WORLD, BOX_SIZE, bc_to_t0, lo, hi, small
	CHARACTER*48	LOHI_STRING

* initialize
	IF (cx .GE. 1) THEN
	   aux_span_compress = cx_unstand_grid(cx)
     .			 .AND. cx_regrid_trans(idim,cx) .EQ. pauxrgrd_pwlave
	ELSE
	   aux_span_compress = .FALSE.
	ENDIF

* grid number is used to determine formatting option
* ( cx<0 for stored (named) regions )  (cx=cx_last for default region)
* 2/93 - treat cx_last like stored regions
* 2/93 - expanded the logic for finding **some** grid to use when their
*        isn't one - search up the context stack
        IF ( cx .LE. 0 .OR. cx .EQ. cx_last ) THEN
           cx_try = is_cx( 1 )
           IF ( cx_try.EQ.0 .OR. cx_try.EQ.unspecified_int4 ) THEN
              grid = unspecified_int4
           ELSE
              grid = cx_grid( cx_try )
           ENDIF
* ... desperation search added 2/3/93
* don't know why is_cx(1) isn'r reliable (havn't checked)
           IF ( grid .EQ. unspecified_int4 ) THEN
              DO 100 cx_try = cx_cmnd+1, cx_cmnd+5   ! arbitrary limit
                 grid = cx_grid( cx_try )
                 IF ( grid.NE.0 .AND. grid.NE.unspecified_int4) GOTO 200
 100          CONTINUE
           ENDIF
        ELSE
           grid = cx_grid( cx )
        ENDIF

* variable/grid from a dsg dataset?  Use the DSG xlate grid.

	its_dsg = TM_ITSA_DSG(grid) .OR. ITSA_DSG_POINTS(cx)

        dset = pdset_irrelevant
	IF (cx .GT. cx_none) dset = cx_data_set(cx)
	IF (dset .GT. pdset_irrelevant) THEN
	   IF ( dsg_ragged(dset) ) THEN
	      IF (dset .EQ. TM_DSG_DSET_FROM_GRID(grid)) 
     .	        grid = dsg_xlate_grid(dset)
	   ENDIF
	ENDIF

	IF ( grid .EQ. unspecified_int4 ) grid = mgrid_abstract

* is this just a single grid box with the world coord box edges given ?
* 1/14: for gz(depth)=0:100@ave we have a valid single point target axis
* on which we want to document the axis limits (in cx_lo/hi_ww), rather
* than the point coordinate.  Look at cx_regrid_trans to decide.
 200	   IF ( cx_lo_ss(cx,idim) .NE. unspecified_int4
     .	.AND.cx_lo_ss(cx,idim) .EQ. cx_hi_ss(cx,idim) 
     .	.AND. .NOT.aux_span_compress                            ) THEN
	   lo = TM_WORLD( cx_lo_ss(cx,idim),grid, idim, box_middle )
	   hi = lo
	ELSE
	   lo = cx_lo_ww( idim, cx )
	   hi = cx_hi_ww( idim, cx )
	ENDIF

	IF (idim.EQ.t_dim .AND. its_dsg .AND. 
     .	    cx_lo_ww( idim,cx ).NE.unspecified_val8) THEN
	   lo = cx_lo_ww( idim, cx )
	   hi = cx_hi_ww( idim, cx )
	ENDIF

* If there are constraints on a DSG dataset, that affects the region in all
* directions, e.g. list/x= /y= affects the time range of data as well.

	IF (its_dsg) THEN
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   CALL MASKED_DSG_RANGE (dset, is_cx(1), idim, lo, hi)
	ENDIF

* hack for E axis positions, so @AVE gets labeled with integer index nos
* *sh* 4/16 - this is a poor place to have fixed this problem
*     better would be to create the concept of an "enumerated axis", which
*     should always be represented as an integer position.  A quick way to
*     do this is to create ITSA_ENUMERATED_AXIS, which (for now) answers .TRUE.
*     only for E axes with units of plag_ax_units_e = 'realization'
*     (This can be expanded later as more enumerated cases arise.)
*     Then in TM_WORLD when returning a coordinate on an enumerated axis ignore
*     box_lo_lim and box_hi_lim, returning box_middle (the integer position)
	its_float = lo .NE. INT(lo)
	IF (its_float .AND. idim.EQ.E_dim) THEN
	   line = grid_line(idim,grid)
	   IF ( (cx .GT. cx_buff)
     .    .AND. line_units(line).EQ.plag_ax_units_e ) THEN
	      cat = cx_category(cx)
	      var = cx_variable(cx) 
	      IF (cx_trans(idim,cx) .GT. trans_compress_code
     .	   .OR.   (cat.EQ.cat_user_var
     .       .AND. SCALAR_GRID_LINE(idim, grid, cat, var)) ) THEN
	         lo = lo + 0.5
	         hi = hi - 0.5
	      ENDIF
	   ENDIF
	ENDIF

* determine precision level for output string
	IF ( full_prec ) THEN
	   ndec = ax_dec_pt( idim )
	ELSE
	   ndec = 0
	ENDIF
	
* What precision is needed to represent this delta and world range ?
* Will write time string according to the resolution of the time axis
* Uses same method as for text listings

	decimal = ndec
	IF (idim.EQ.t_dim .AND. ABS(ndec).GT.6) THEN
	   loss = cx_lo_ss(cx,idim)
	   hiss = cx_hi_ss(cx,idim)
	   delss = 1
	   IF (loss .NE. unspecified_int4) delss = cx_delta (idim,cx)

* how large is the smallest "delta" world coordinate to represent ?
	   small = +1.E33
	   IF (line_delta(grid_line( idim, grid )) .NE. unspecified_val8) THEN
              small = line_delta( grid_line(idim, grid) )
	   ELSE
 	      DO j = cx_lo_ss(cx,idim), cx_hi_ss(cx,idim), delss
	         small = MIN( small, BOX_SIZE( j, grid, idim ) )
 	      ENDDO
	   ENDIF

	   decimal = DATE_PREC( grid, idim, small )
	   decimal = decimal + 1
	ENDIF

* create the string
	LOHI_WORLD = LOHI_STRING( lo,hi,idim,grid,to_word,decimal,slen )
	
	RETURN
	END
