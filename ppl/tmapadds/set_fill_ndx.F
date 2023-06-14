	SUBROUTINE SET_FILL_NDX (fill_ndx)


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
*	Sets fill index appropriately.  DEC uses bundled indices,
*	at this date ATC has too few to use them, so must use individual
*	attributes.

*	J Davison 10.12.92
*       *jd* 7.31.97 mode to change call to gqlwk to query_xgks_wkstn
*       *jd* 10.98 handle "atomic" colors
* 3/99 *kob* 	move var init off of decl list  - cleaning up after jerry.

        IMPLICIT NONE
        include         'shade_vars.cmn'
	include		'gkscm1_inc.decl'
	include		'GKSCM1.INC'
        include        'gkspar.inc'

	integer		fill_ndx,style,pattern_ndx,solid
	DATA            solid/21/
        integer         max_pline,max_pmark,max_text,max_fill_area
        integer         max_pattern,max_color,error,color_ndx
	real		incr,spc,min_spc,delta_spc
  
	parameter	(min_spc=0.05,delta_spc=0.01)

*******************************************************************************

	if (area_bundles) then
*	   USE SET FILL AREA INDEX
	   call FGD_GSFAI(fill_ndx+lo_color_ndx(wsid))
	else
*	   HOW MANY COLORS HAVE WE?
           call FGD_GQLWK (wstype,error,max_pline,max_pmark,
     .              max_text,max_fill_area,max_pattern,max_color)

	   if (max_color .gt. 2) then 
*	      CAN STILL USE COLOR INDICES -- BUT INDIVIDUAL
	      
*             USE ATOMIC COLORS?
	      if (rgb_mapping .ne. by_level) then
*                SMOOTH OUT COLORS TO FULL RANGE
	         call FGD_GSFACI(fill_ndx+lo_color_ndx(wsid)-2)
	      else
*                USE ATOMIC COLORS -- REUSE AS NECESSARY
		 color_ndx = mod((fill_ndx - 2),ncl) + 2
		 call FGD_GSFACI(color_ndx+lo_color_ndx(wsid)-2)
	      end if
	   endif
	endif

	pattern_ndx = mod((fill_ndx - 2),npat) + 1

	if (patterns(pattern_ndx) .eq. solid) then
	   call FGD_GSFAIS(GSOLID)
	else
	   call FGD_GSFAIS(GHATCH)
	   call FGD_GSFASI(-1 * patterns(pattern_ndx))
	end if

	return
	end

