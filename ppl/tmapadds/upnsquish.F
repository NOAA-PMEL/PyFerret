
	SUBROUTINE UPNSQUISH (instr,outstr,len_out)

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
*	Routine to convert variable length string to upper case
*	and left justify, throwing away all blanks, and returning 
*	the length of the resulting string.  

	integer		len_in,len_out,i
	character*(*)	instr,outstr

	integer		mask
	data		mask/Z'DF'/

********************************************************************************

	len_in = len (instr)
	outstr = ' '

	len_out = 0
	do 10 i = 1,len_in
	   if (instr(i:i) .eq. ' ') goto 10

	   len_out = len_out + 1
	   if (instr(i:i) .lt. 'a' .or. instr(i:i) .gt. 'z' ) then     
	      outstr(len_out:len_out) = instr(i:i)
           else                                                         
#ifdef unix

#ifdef NEED_IAND
#ifdef FORTRAN_90
C	*jd* Mod for linux port *jd* 12.96
	outstr(len_out:len_out) = 
     x	char(iand(mask,ichar(instr(i:i))))
#else
	outstr(len_out:len_out) = 
     x	char(iand('DF'x,ichar(instr(i:i))))
#endif
#else
              outstr(len_out:len_out) = 
     .              char(and('DF'x,ichar(instr(i:i))))
#endif
#else
              outstr(len_out:len_out) = 
     .              char(iand('DF'x,ichar(instr(i:i))))
#endif
           end if      
10	continue

	return
	end
