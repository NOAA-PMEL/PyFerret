	SUBROUTINE PROCESS_X_EVENTS

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



