        CHARACTER*8 FUNCTION EXPR_NAME( n )

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
* create a unique name for a user-defined variable
* "expr#1","expr#2",..."expr#9","expr#:","expr#,",etc. from ASCII table

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/23/87
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *kms* 7/18 revise method for generating the name to just use numbers;
*                  original method gives lowercase characters ('a' at n=49)
*                  that match earlier uppercase characters ('A' at n=17),
*                  and would blow up if n exceeds 207.

        IMPLICIT NONE
        INTEGER n, k
        CHARACTER*8 namecode

! original method:
!        EXPR_NAME = 'EX#'//CHAR(48+n)

        IF ( n .LT. 0 ) THEN
            STOP 'EXPR_NAME: unsupported request ' //
     .           'for name code with n < 0'
        ELSEIF ( n .LT. 10 ) THEN
  10        FORMAT('EX#',I1)
            WRITE(namecode, 10) n
        ELSEIF ( n .LT. 100 ) THEN
  20        FORMAT('EX#',I2)
            WRITE(namecode, 20) n
        ELSEIF ( n .LT. 1000 ) THEN
  30        FORMAT('EX#',I3)
            WRITE(namecode, 30) n
        ELSEIF ( n .LT. 10000 ) THEN
  40        FORMAT('EX#',I4)
            WRITE(namecode, 40) n
        ELSEIF ( n .LT. 100000 ) THEN
  50        FORMAT('EX#',I5)
            WRITE(namecode, 50) n
        ELSE
            STOP 'EXPR_NAME: unsupported request ' //
     .           'for name code with n > 99999'
        ENDIF

        EXPR_NAME = namecode

        RETURN
        END

