	SUBROUTINE SETUP_KEYS( nvar )

* determine parameters needed to layout titles and keys on the plot based on
* how many variables are in the plot

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200: 12/21/89
* 	  5/6/90 - fixed bug when xlen is very small using MAX
* V420:  4/23/96 - number of columns was computed without respect to nvar
*	  	thus if the plot could fit 3 keys across but only 2 were needed
*	  	they didn't end up centered

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/plot_setup.parm'
	include	'ferret_cmn/command.parm'
	include	'ferret_cmn/xplot_setup.cmn'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include 'tmap_pplv11inc/axis_inc.decl'		! axis lengths
	include 'pplv11inc/AXIS.INC'		! axis lengths
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE	'FERRET_CMN:COMMAND.PARM'
	INCLUDE	'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'		! axis lengths
	INCLUDE 'PPLV11INC:AXIS.INC'		! axis lengths
#endif

* calling argument declarations
	INTEGER	nvar

* local variable declarations:
	LOGICAL	change_yorg, change_ysize
	INTEGER	nrow, idim
	REAL	ysqueeze, yspace_needed, yspace_avail

* how many columns of keys can fit on the plot ?
	key_ncolmn  = MAX( 1, INT((xlen - key_xorigin)/key_xspace0) )
	IF (key_ncolmn .GT. nvar) key_ncolmn = nvar	! 4/96
	key_xspace  = MAX( xlen/5., (xlen - key_xorigin) / key_ncolmn )
	key_xsize   = MAX( xlen/6., key_xspace - key_xwhite_space )

* and how many rows do we need ?
	nrow = ((nvar-1) / key_ncolmn) + 1
	yspace_needed = nrow*key_yspace0

* where was the y origin before we tampered with it ?
	IF ( saved_ppl_yax ) THEN
	   yspace_avail = ppl_saved_yorg
	ELSE
	   yspace_avail = yorg
	ENDIF
	yspace_avail = yspace_avail + key_yorigin

* compute how much we will allow the key labels to shrink
	IF ( nrow .LE. 2 ) THEN
	   ysqueeze = 0.9
	ELSE
	   ysqueeze = MAX( 0.75, 0.92**(nrow-1) )
	ENDIF

* determine if
*  1) the keys will fit using the default y origin and key label size
*  2) the keys will fit but only by shrinking the label size
*  3) the keys will fit only by shifting the y origin (y location of x axis)
	IF ( cmnd_num .NE. cmnd_plot ) THEN
	   change_ysize = .FALSE.
	   change_yorg  = .FALSE.
	ELSE
	   change_ysize = yspace_avail .LT. yspace_needed
	   change_yorg  = yspace_avail .LT. yspace_needed*ysqueeze
	ENDIF

	IF ( .NOT. change_yorg ) THEN
* ... restore the default vertical layout if we tampered with it
           IF ( saved_ppl_yax ) THEN
	      ylen = ppl_saved_yaxlen
	      yorg = ppl_saved_yorg
	      saved_ppl_yax = .FALSE.
	   ENDIF
	ENDIF

	IF ( change_yorg ) THEN
* ... shrunken keys AND increased yorigin
* ... save the default vertical layout so we can restore it later
	   IF ( .NOT. saved_ppl_yax ) THEN
	      ppl_saved_yorg   = yorg
	      ppl_saved_yaxlen = ylen
	      saved_ppl_yax = .TRUE.
	   ENDIF 
	   yorg = yspace_needed * ysqueeze - key_yorigin
	   ylen = (ppl_saved_yaxlen + ppl_saved_yorg) - yorg
	   key_yspace = key_yspace0 * ysqueeze
	   key_ysize  = (key_yspace0 - key_ywhite_space) * ysqueeze
	   key_labht  = key_labht0 * ysqueeze
	ELSEIF ( change_ysize ) THEN
* ... shrunken keys but don't tamper with y origin
	   ysqueeze = yspace_avail / yspace_needed
	   key_yspace = key_yspace0 * ysqueeze
	   key_ysize  = (key_yspace0 - key_ywhite_space) * ysqueeze
	   key_labht  = key_labht0 * ysqueeze
	ELSE
* ... full-sized keys, don't tamper with y origin
	   key_yspace = key_yspace0
 	   key_ysize  = key_yspace0 - key_ywhite_space
	   key_labht  = key_labht0
	ENDIF

* how many labels in the upper left corner?
	ul_nlabs = 0
	DO 100 idim = 1, 4
	   ul_dolab(idim) = cx_lo_ww(idim,cx_plot).NE.unspecified_val8
     . 		      .AND. cx_lo_ww(idim,cx_plot).NE.unspec_val8_2
	   IF ( ul_dolab(idim) ) ul_nlabs = ul_nlabs + 1	
 100	CONTINUE	

	RETURN
	END
