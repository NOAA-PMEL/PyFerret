	SUBROUTINE TM_PACK_STRING(in_string,out_string,start,end,len_out)

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
* remove blanks and tabs from the designated portion of the string

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 03/04/87
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an character constant instead
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

*
* calling argument declarations:
	INTEGER		start, end, len_out
	CHARACTER*(*)	in_string, out_string

* internal parameter declarations:
	CHARACTER*1	tab
	PARAMETER     ( tab = CHAR(9))


* internal variable declarations:
	INTEGER	i_in, i_out, str_len, str_end

* true string length
	str_len = MIN( LEN(in_string), LEN(out_string) )
	str_end = MIN( str_len, end )

* transfer all characters until "start" to output string
	DO i_in = 1, start
 	   out_string( i_in:i_in ) = in_string( i_in:i_in )
	ENDDO
* pack the indicated region
	i_out = start

	DO 100 i_in = start, str_end
* ... skip hole ?
	   IF ( in_string( i_in:i_in ) .EQ. ' '
     .	   .OR. in_string( i_in:i_in ) .EQ. tab ) GOTO 100

* ... transfer character to output
	   out_string( i_out:i_out ) = in_string( i_in:i_in )
	   i_out = i_out + 1
 100	CONTINUE
	   
* transfer the remainder to the output
	DO 200 i_in = str_end+1, str_len
	   out_string( i_out:i_out ) = in_string( i_in:i_in )
	   i_out = i_out + 1
 200	CONTINUE

* length of packed output string
	len_out = i_out - 1

* blank out old shifted characters at right hand end
	DO i_out = i_out, str_len
 	   out_string(i_out:i_out) = ' '
	ENDDO

	RETURN
	END
