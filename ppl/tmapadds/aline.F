	SUBROUTINE ALINE (x,nxs)

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
*	Processes ALINE command buffer and manages aline database
*	J Davison 10.16.89
*	11.9.93 Mod to put immediate mode output in segments
* v552 *acm* 3/03 Longer strings: changes for multi-line labels.

	include 	'parampl5_dat.decl'            
	include 	'PARAMPL5.DAT'            
	include 	'cmrd_inc.decl'            
	include 	'CMRD.INC'            
	include 	'cmrdl_inc.decl'            
	include 	'CMRDL.INC'            
	include 	'lines_inc.decl'            
	include 	'LINES.INC'            
	include 	'aline.cmn'            
	include 	'switch_inc.decl'            
	include 	'SWITCH.INC'            

* special COMMON to let this routine know if FERRET is in control
        include 'ppl_in_ferret.cmn'
  
	integer		nxs,end,a_line,i
	real		x(nxs),minx,miny,maxx,maxy
	character*2048	str

********************************************************************************

	call upnsquish (label,str,end)
                                
*	IF NO LABEL, EXECUTE ALINE COMMAND IMMEDIATELY
	if (end .eq. 0) then

*	   CANCEL ALL SET LINES IF COMMAND IS SIMPLY 'ALINE'
	   if (m .eq. 0) then
	      do 10 i = 1,nlines
		 aline_on(i) = .false.
10	      continue
	   end if 

*	   OTHERWISE REQUIRE ALL ARGUMENTS   
	   if (m .ne. 5) return
	   
	   if (p(1) .gt. 0 .and. p(1) .le. nlines) then
	      a_line = p(1)
	   else                   
	      a_line = 1               
	   end if 

*	   HANDLE /USER & /NOUSER COORDINATES PROPERLY
	   if (suser) then
	      minx = p(2)
	      miny = p(3)
	      maxx = p(4)
	      maxy = p(5)
	   else
	      call point_convert (p(2),p(3),minx,miny)
	      call point_convert (p(4),p(5),maxx,maxy)
	   end if

!          Mod 11.9.93 to insert aline immediate output into segments
*          TMAP check that output window is active
           IF ( PPL_in_FERRET ) CALL CK_GKS_ACTIVE( *9999 )
*          TMAP addition to turn on segment
           IF ( PPL_in_FERRET ) CALL SEG_ON( )

	   call aline_sub (x,nxs,a_line,minx,miny,maxx,maxy)
           IF ( PPL_in_FERRET ) CALL SEG_OFF
9999	   return
	end if

*	IF LABEL = 'ON' THEN SET ALINE FOR THIS LINE#                          
	if (str(1:2) .eq. 'ON') then
	   if (m .eq. 5 .and. p(1) .gt. 0 .and. p(1) .le. nlines) then
	      a_line			= p(1)
	      aline_on(a_line)		= .true.
	      aline_user(a_line)	= suser
	      aline_minx(a_line)	= p(2)
	      aline_miny(a_line)	= p(3)
	      aline_maxx(a_line)	= p(4)
	      aline_maxy(a_line)	= p(5)
	   end if
	   return
	end if

*	IF LABEL = 'OFF' THEN CANCEL ALINE FOR THIS LINE#
	if (str(1:3) .eq. 'OFF') then
	   a_line 			= p(1)
	   aline_on(a_line)		= .false.
	end if

	return

	end


