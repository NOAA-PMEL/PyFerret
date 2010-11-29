	BLOCK DATA ENCODE_STRINGS_DATA


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
*
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
* 11/2009


	INCLUDE 'encode_strings.cmn'

* Reserved chars to encode

	DATA reschar(1)  /'!'/
	DATA reschar(2)  / '*'/
	DATA reschar(3)  / '"'/
	DATA reschar(4)  / ''''/
	DATA reschar(5)  / '('/
	DATA reschar(6)  / ')'/
	DATA reschar(7)  / ';'/
	DATA reschar(8)  / ':'/
	DATA reschar(9)  / '@'/
	DATA reschar(10) / '&'/
	DATA reschar(11) / '+'/
	DATA reschar(12) / '$'/
	DATA reschar(13) / ','/
	DATA reschar(14) / ','/
	DATA reschar(15) / '?'/
	DATA reschar(16) / '%'/
	DATA reschar(17) / '#'/
	DATA reschar(18) / '['/
	DATA reschar(19) / ']'/
	DATA reschar(20) / ' '/
	DATA reschar(21) / '{'/
	DATA reschar(22) / '}'/
	DATA reschar(23) / '='/
	DATA reschar(24) / '`'/
	

** examples break when I encode a /
**	DATA reschar(14) / '/'/
**	DATA subencode(14) /'%2F'/

* The corresponding encoding
	DATA subencode(1)  /'%21'/
	DATA subencode(2)  /'%2A'/
	DATA subencode(3)  /'%22'/
	DATA subencode(4)  /'%27'/
	DATA subencode(5)  /'%28'/
	DATA subencode(6)  /'%29'/
	DATA subencode(7)  /'%3B'/
	DATA subencode(8)  /'%3A'/
	DATA subencode(9)  /'%40'/
	DATA subencode(10) /'%26'/
	DATA subencode(11) /'%2B'/
	DATA subencode(12) /'%24'/
	DATA subencode(13) /'%2C'/
	DATA subencode(14) /'%2C'/
	DATA subencode(15) /'%3F'/
	DATA subencode(16) /'%25'/
	DATA subencode(17) /'%23'/
	DATA subencode(18) /'%5B'/
	DATA subencode(19) /'%5D'/
	DATA subencode(20) /'%20'/
	DATA subencode(21) /'%7B'/
	DATA subencode(22) /'%7D'/
	DATA subencode(23) /'%3D'/
	DATA subencode(24) /'%60'/

	DATA nsubencode /nencode*3/

	END
