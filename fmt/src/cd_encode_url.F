	SUBROUTINE CD_ENCODE_URL (urlstring)
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
* Encode a URL that has an expression, which is set off by the string
* _expr_{}. (_expr_ is the "marker" string - could be something else.) 
* Encode everything starting with the {} after _expr_ and up 
* to and including the next }.  Throw away anything else which would be 
* a constraint (We may deal with constraints later...)

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
* 10/2009

*Reserved characters after percent-encoding ; the last being a space, is %20.
*
*      !  %21
*      *  %2A
*      "  %22
*      '  %27
*      (  %28
*      )  %29
*      ;  %3B
*      :  %3A
*      @  %40
*      &  %26
*      =  %3D
*      +  %2B
*      $  %24
*      ,  %2C
*      /  %2F
*      ?  %3F
*      %  %25
*      #  %23
*      [  %5B
*      ]  %5D
*         %20
*      {  %7B
*      }  %7D
*

	INCLUDE 'xrisc_buff.cmn'
	INCLUDE 'encode_strings.cmn'
	
	EXTERNAL ENCODE_STRINGS_DATA

* Argument declarations
	CHARACTER*(*) urlstring

* Local declarations

	INTEGER TM_LENSTR1, slen, mlen, blen, iencode, loc, bracket
	CHARACTER*10 marker

* Save dataset info in GLOBAL_NCdsetList for attribute handling

	marker = '_expr_{}'

	slen = TM_LENSTR1( urlstring )
	mlen = TM_LENSTR1(marker)

        iencode = INDEX( urlstring(1:slen), marker(1:mlen) ) 

* Encode an F-TDS expression?

        IF (iencode .EQ. 0) RETURN

	loc = iencode + mlen - 3  ! up to the {}
        risc_buff = urlstring(1:loc)
	blen = loc

        loc = loc + 1
	bracket = 0

	DO WHILE (loc .LE. slen)
	   DO 100 i = 1, nencode
	      IF (urlstring(loc:loc) .EQ. reschar(i)) THEN
	         risc_buff = risc_buff(1:blen)//subencode(i)
		 blen = blen + nsubencode(i)
		 GOTO 200
	      ENDIF
  100      CONTINUE

*  Not encoded, just keep the orig. character

           risc_buff = risc_buff(1:blen)//urlstring(loc:loc)
	   blen = blen + 1
  200      CONTINUE
 
* Stop the second time a } is reached; this is the end of the Ferret expr.
* (If there is a DODS constraint on the orig. URL, would it also get encoded?)

	   IF (urlstring(loc:loc) .EQ. '}') THEN
	      IF (bracket .GT. 0) GOTO 300
	      bracket = bracket + 1
	   ENDIF

           loc = loc + 1

	ENDDO

  300   CONTINUE

        urlstring = risc_buff(1:blen)
	print *, urlstring(1:blen)
	
	RETURN
	END

