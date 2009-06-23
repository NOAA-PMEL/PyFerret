	INTEGER FUNCTION GET_NDX (datum,ndx)

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
*	Gets bundle index for fill area calls
*	J Davison 5.17.90
*	Mod *jd* 10.1.92 to use 1 fill index                    

* 	Mod *sh* 4/97 -- removed checks on valid coordinates from this routine
*		to a separate new routine, BAD_XY_COORD

* 	Mod *sh* 11/97 -- changed calling args by eliminating "i,j" so this
*		routine can be used for 1D (polygon fill) as well as 2D
*		Also: bug in this code: if there is no valid
*	        index for datum then "bad_val" was returned. But bad_val is
*		a REAL!!   Now return -999 instead (quick hack).
*		Also had to add explicit range checks since whatever logic
*		made rng chks mysteriously unnecessary using REAL bad_flag has
*		changed
*  V5.71 *acm*  6/15/04 Use TM_FPEQ when checking whether a single-level zlev is
*               the same as datum.
*  v581 *acm* 3/05 open levels (-INF) (INF)
*  V62+ *acm* 6/09 fix bug 1671: data level exactly matching top of upper level 
*                  wasnt plotted on SHADE and POLY plots

	include		'parampl5_dat.decl'
	include		'PARAMPL5.DAT'
	include		'miss_inc.decl'
	include		'MISS.INC'
        include         'hd_inc.decl'
        include         'HD.INC'
	include		'cont_inc.decl'
	include		'CONT.INC'
	include		'shade_vars.cmn'

* Calling argument declarations
	integer		ndx
        real            datum

* Local variable declarations
	real	        test
	integer		lev1, levn
        logical         TM_FPEQ

********************************************************************************

*	      CHECK AGAINST 'LIMITS' VALUES AND FIND COLOR INDEX

              lev1 = 1
              levn = nlev2
              IF (neginf) lev1 = 2
              IF (posinf) levn = nlev2 - 1

	      if(
     .		 (zle.and.datum.le.cmzle) 	.or.
     .		 (zeq.and.datum.eq.cmzeq) 	.or.
     .		 (zge.and.datum.ge.cmzge)) then

	         get_ndx = -999
* ... 12/4/97 added range checks

	      else if (datum .lt. zlev(lev1) .AND. (.NOT.neginf) ) then
	         get_ndx = -999
	      else if (datum .gt. zlev(levn) .AND. (.NOT.posinf) ) then
	         get_ndx = -999
* ... 12/4/97

c	      else if (levn - lev1 .LE. 1) then
	      else if (levn .LE. 1) then
		 if ( TM_FPEQ(datum, zlev(lev1)) ) then
		    get_ndx = 2
		 else
		    get_ndx = -999
		 endif
	      else   
                 IF (( datum .LT. zlev(lev1) ) .AND. neginf ) THEN
                    get_ndx = 2
                 ELSE IF ( (datum .GT. zlev(levn) ) .AND. posinf) THEN
                    get_ndx = nlev2
                 ELSE
	            call locator (zlev(lev1),lev1,levn,datum,ndx)

	            if (ndx .eq. levn) THEN
		       IF (datum .eq. zlev(levn)) then 
		          get_ndx = levn
		       else
                          IF (posinf) then
	                     get_ndx = nlev2
	                  else
                             get_ndx = -999
			  endif
                       endif
	            else
	               get_ndx = ndx + 1
	            end if
                 ENDIF
	      end if

	return
	end
