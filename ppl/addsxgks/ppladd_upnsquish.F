
	SUBROUTINE UPNSQUISH (instr,outstr,len_out)

*	J Davison 3.15.89 JISAO/PMEL/TMAP
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
