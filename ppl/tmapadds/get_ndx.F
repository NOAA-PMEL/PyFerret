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

#ifdef unix
	include		'tmap_pplv11inc/parampl5_dat.decl'
	include		'pplv11inc/PARAMPL5.DAT'
	include		'tmap_pplv11inc/miss_inc.decl'
	include		'pplv11inc/MISS.INC'
        include         'tmap_pplv11inc/hd_inc.decl'
        include         'pplv11inc/HD.INC'
	include		'tmap_pplv11inc/cont_inc.decl'
	include		'pplv11inc/CONT.INC'
	include		'tmap_pplv11inc/shade_vars.cmn'

#else
	INCLUDE		'tmap_pplv11inc:tmap_parampl5.dat'
	INCLUDE		'tmap_pplv11inc:tmap_miss.inc'
        INCLUDE         'tmap_pplv11inc:tmap_hd.inc'
	INCLUDE		'tmap_pplv11inc:tmap_cont.inc'
	INCLUDE		'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

	integer		ndx

	real		datum

********************************************************************************

*	      CHECK AGAINST 'LIMITS' VALUES AND FIND COLOR INDEX
	      if(
     .		 (zle.and.datum.le.cmzle) 	.or.
     .		 (zeq.and.datum.eq.cmzeq) 	.or.
     .		 (zge.and.datum.ge.cmzge)) then

	         get_ndx = -999
* ... 12/4/97 added range checks
	      else if (datum .lt. zlev(1)) then
	         get_ndx = -999
	      else if (datum .gt. zlev(nlev2)) then
	         get_ndx = -999
* ... 12/4/97

	      else if (nlev2 .eq. 1) then
		 if (datum .eq. zlev(1)) then
		    get_ndx = 2
		 else
		    get_ndx = -999
		 endif
	      else             
	         call locator (zlev,1,nlev2,datum,ndx)

	         if (ndx .eq. nlev2) then
	            get_ndx = -999
	         else
	            get_ndx = ndx + 1
	         end if
	      end if

	return
	end
