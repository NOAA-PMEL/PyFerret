	SUBROUTINE PROCESS_X_EVENTS

*	Update ATC GKS workstations if necessary.  First attempt to
*	determine if this works and is a bit heavy handed.

*	J Davison 4.29.92
*	Mod *jd* 5.26.92 to include -5399 escape function
*       Mod *jd* 3.7.94  to include xgks event handling, on Solaris.  Renamed
*	from atc_update to process_x_events

#ifdef unix
        include         'tmap_pplv11inc/gkscm2.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

#ifdef atc_gks
	integer 	i,err,num_open,wsid,num
        integer 	ierr, ia(3), lidr, lodr
        real		ra(1)
        character*80 	str(1), datarec(1), odr(1)

#   ifdef unix
        include        'atc_gksdir/gkspar.inc'
#   else
        INCLUDE        'atc_gksdir:gkspar.inc'
#   endif

******************************************************************************

	if (.not. gks_open) goto 1000

	call gqopwk (0,err,num_open,wsid)
	if (err. ne. 0) goto 1000

	do 10 i = 1,num_open
	   call gqopwk (i,err,num,wsid)
	   if (err. ne. 0) goto 10

           ia(1) = wsid
           call gprec (1,ia,0,ra,0,0,str,80,ierr,lidr,datarec)
           call guesc (-5399,lidr,datarec,80,lodr,odr)
10	continue

1000	return

#endif

#ifdef xgks
	if (gks_open) call xgks_x_events
#endif

	end



