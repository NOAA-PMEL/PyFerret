	SUBROUTINE SET_DEFAULT_COLORS (zwsid,zwstype,ndx_lo,range)
	
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
*	Sets the color map between ndx_lo and ndx_hi to ws default
*	J Davison JISAO/PMEL/TMAP 8.1.89

#ifdef unix
        include         'shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

	integer		zwsid,ndx_lo,range,ndx_hi,zwstype,err,ndx
	real		r_val,g_val,b_val

********************************************************************************

	ndx_hi	= ndx_lo + range - 1

*	SET ALL COLORS TO THE DEFAULT
	do 50 ndx = ndx_lo,ndx_hi

*	   WHAT IS THE DEFAULT COLOR FOR EACH NDX?
	   call gqpcr (zwstype,ndx,err,r_val,g_val,b_val)
	   if (err .ne. 0) goto 50

*	   SET THE COLOR OF THIS NDX
	   call gscr (zwsid,ndx,r_val,g_val,b_val)
50	end do

	return
	end
