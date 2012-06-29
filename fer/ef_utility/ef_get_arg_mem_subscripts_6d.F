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
* Return the array memory limits in the common blocks for those
* times when they cannot be retrieved directly.
*
      SUBROUTINE EF_GET_ARG_MEM_SUBSCRIPTS_6D(id, arg_memlo, arg_memhi)

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'EF_Util.parm'
      INCLUDE 'EF_mem_subsc.cmn'

* argument declarations
      INTEGER id, arg_memlo(nferdims,EF_MAX_ARGS),
     .            arg_memhi(nferdims,EF_MAX_ARGS)

* this assumes nferdims .EQ. 6 and EF_MAX_ARGS .EQ. 9
      arg_memlo(1,1) = mem1lox
      arg_memlo(1,2) = mem2lox
      arg_memlo(1,3) = mem3lox
      arg_memlo(1,4) = mem4lox
      arg_memlo(1,5) = mem5lox
      arg_memlo(1,6) = mem6lox
      arg_memlo(1,7) = mem7lox
      arg_memlo(1,8) = mem8lox
      arg_memlo(1,9) = mem9lox

      arg_memlo(2,1) = mem1loy
      arg_memlo(2,2) = mem2loy
      arg_memlo(2,3) = mem3loy
      arg_memlo(2,4) = mem4loy
      arg_memlo(2,5) = mem5loy
      arg_memlo(2,6) = mem6loy
      arg_memlo(2,7) = mem7loy
      arg_memlo(2,8) = mem8loy
      arg_memlo(2,9) = mem9loy

      arg_memlo(3,1) = mem1loz
      arg_memlo(3,2) = mem2loz
      arg_memlo(3,3) = mem3loz
      arg_memlo(3,4) = mem4loz
      arg_memlo(3,5) = mem5loz
      arg_memlo(3,6) = mem6loz
      arg_memlo(3,7) = mem7loz
      arg_memlo(3,8) = mem8loz
      arg_memlo(3,9) = mem9loz

      arg_memlo(4,1) = mem1lot
      arg_memlo(4,2) = mem2lot
      arg_memlo(4,3) = mem3lot
      arg_memlo(4,4) = mem4lot
      arg_memlo(4,5) = mem5lot
      arg_memlo(4,6) = mem6lot
      arg_memlo(4,7) = mem7lot
      arg_memlo(4,8) = mem8lot
      arg_memlo(4,9) = mem9lot

      arg_memlo(5,1) = mem1loe
      arg_memlo(5,2) = mem2loe
      arg_memlo(5,3) = mem3loe
      arg_memlo(5,4) = mem4loe
      arg_memlo(5,5) = mem5loe
      arg_memlo(5,6) = mem6loe
      arg_memlo(5,7) = mem7loe
      arg_memlo(5,8) = mem8loe
      arg_memlo(5,9) = mem9loe

      arg_memlo(6,1) = mem1lof
      arg_memlo(6,2) = mem2lof
      arg_memlo(6,3) = mem3lof
      arg_memlo(6,4) = mem4lof
      arg_memlo(6,5) = mem5lof
      arg_memlo(6,6) = mem6lof
      arg_memlo(6,7) = mem7lof
      arg_memlo(6,8) = mem8lof
      arg_memlo(6,9) = mem9lof

      arg_memhi(1,1) = mem1hix
      arg_memhi(1,2) = mem2hix
      arg_memhi(1,3) = mem3hix
      arg_memhi(1,4) = mem4hix
      arg_memhi(1,5) = mem5hix
      arg_memhi(1,6) = mem6hix
      arg_memhi(1,7) = mem7hix
      arg_memhi(1,8) = mem8hix
      arg_memhi(1,9) = mem9hix

      arg_memhi(2,1) = mem1hiy
      arg_memhi(2,2) = mem2hiy
      arg_memhi(2,3) = mem3hiy
      arg_memhi(2,4) = mem4hiy
      arg_memhi(2,5) = mem5hiy
      arg_memhi(2,6) = mem6hiy
      arg_memhi(2,7) = mem7hiy
      arg_memhi(2,8) = mem8hiy
      arg_memhi(2,9) = mem9hiy

      arg_memhi(3,1) = mem1hiz
      arg_memhi(3,2) = mem2hiz
      arg_memhi(3,3) = mem3hiz
      arg_memhi(3,4) = mem4hiz
      arg_memhi(3,5) = mem5hiz
      arg_memhi(3,6) = mem6hiz
      arg_memhi(3,7) = mem7hiz
      arg_memhi(3,8) = mem8hiz
      arg_memhi(3,9) = mem9hiz

      arg_memhi(4,1) = mem1hit
      arg_memhi(4,2) = mem2hit
      arg_memhi(4,3) = mem3hit
      arg_memhi(4,4) = mem4hit
      arg_memhi(4,5) = mem5hit
      arg_memhi(4,6) = mem6hit
      arg_memhi(4,7) = mem7hit
      arg_memhi(4,8) = mem8hit
      arg_memhi(4,9) = mem9hit

      arg_memhi(5,1) = mem1hie
      arg_memhi(5,2) = mem2hie
      arg_memhi(5,3) = mem3hie
      arg_memhi(5,4) = mem4hie
      arg_memhi(5,5) = mem5hie
      arg_memhi(5,6) = mem6hie
      arg_memhi(5,7) = mem7hie
      arg_memhi(5,8) = mem8hie
      arg_memhi(5,9) = mem9hie

      arg_memhi(6,1) = mem1hif
      arg_memhi(6,2) = mem2hif
      arg_memhi(6,3) = mem3hif
      arg_memhi(6,4) = mem4hif
      arg_memhi(6,5) = mem5hif
      arg_memhi(6,6) = mem6hif
      arg_memhi(6,7) = mem7hif
      arg_memhi(6,8) = mem8hif
      arg_memhi(6,9) = mem9hif

      RETURN
      END
