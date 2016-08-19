	CHARACTER*1 FUNCTION UPPER_CASE ( input_char )

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
* convert character string to upper case

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/4/86
* Unix/RISC port - 2/13/91 - .AND. always produces a logical result

* 5/95 *sh* - added ifdef for HP port
* 12/96 *kob* - Linux port 
*               - had to declare hex constants as implicit
*                 integers in DATA statements.  Added a 
*                 FORTRAN_90 ifdef to accomplish this. 
*               - Needs to use IAND rather than AND
* V63  *acm* 10/09 Changes for gfortran build


	CHARACTER*1 input_char

* kob 12/96
#ifdef FORTRAN_90
      INTEGER HEXDF
      DATA HEXDF/Z'DF'/
#endif

	IF ( input_char .LT. 'a' .OR. input_char .GT. 'z' ) THEN
	   upper_case = input_char
	ELSE
* modify if logic for linux - also vms needs iand - like that
* matters anymore - just to keep it clean *kob* 12/96

#if defined unix  && !defined NEED_IAND
           upper_case = CHAR( AND('DF'X , ICHAR( input_char ) ) )
#else
#  ifdef FORTRAN_90 			
	   upper_case = CHAR( IAND(HEXDF, ICHAR( input_char ) ) )
# elif defined gfortran
           upper_case = CHAR( AND('DF'X , ICHAR( input_char ) ) )
#  else
	   upper_case = CHAR( IAND('DF'X , ICHAR( input_char ) ) )
#  endif
#endif
	ENDIF

	RETURN
	END
