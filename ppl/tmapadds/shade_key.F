        SUBROUTINE SHADE_KEY
        
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
*       Sets the options for SHADE key -- size,location,etc
*       J Davison JISAO/PMEL/TMAP 8.24.88

#ifdef unix
        include         'shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

#ifdef unix
	include		'cmrd_inc.decl'
	include		'CMRD.INC'
        include         'axis_inc.decl'
        include         'AXIS.INC'
        include         'plt_inc.decl'
        include         'PLT.INC'
#else
	INCLUDE		'tmap_pplv11inc:tmap_cmrd.inc'
        INCLUDE         'tmap_pplv11inc:tmap_axis.inc'
        INCLUDE         'tmap_pplv11inc:tmap_plt.inc'
#endif

*******************************************************************************

	if (m .ge. 1 .and. iflg(1) .ne. 0 .and. p(1).eq.0.0) 
     .	do_key = .false.
	if (m .ge. 1 .and. iflg(1) .ne. 0 .and. p(1).eq.1.0) 
     .	do_key = .true.

	if (m .ge. 2 .and. iflg(2) .ne. 0 .and. p(2).eq.0.0) 
     .	vertical_key = .false.
	if (m .ge. 2 .and. iflg(2) .ne. 0 .and. p(2).eq.1.0) 
     .	vertical_key = .true.

	if (m .ge. 3 .and. iflg(3) .ne. 0) klab_size = p(3)
	if (m .ge. 4 .and. iflg(4) .ne. 0) klab_incr = ifix (p(4))
	if (m .ge. 5 .and. iflg(5) .ne. 0) klab_dig  = ifix (p(5))
	if (m .ge. 6 .and. iflg(6) .ne. 0) klab_len  = ifix (p(6))

	if (m .ge. 10 .and. iflg(7) .ne. 0 .and. iflg(8) .ne. 0 
     .	.and. iflg(9) .ne. 0 .and. iflg(10) .ne. 0) then
	   kuser_loc = .true.
	   kx_lo = p(7) * 1000.0
	   kx_hi = p(8) * 1000.0
	   ky_lo = p(9) * 1000.0
	   ky_hi = p(10)* 1000.0
        else
	   kuser_loc = .false.
        end if

	return
	end
