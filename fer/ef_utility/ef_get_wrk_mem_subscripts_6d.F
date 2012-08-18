*
* ef_get_wrk_mem_subscripts_6d.F (was ef_get_wrk_mem_subscripts.F)
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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
*  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
* Return the work array memory limits in the common blocks
* for those times when they cannot be retrieved directly.
*
      SUBROUTINE EF_GET_WRK_MEM_SUBSCRIPTS_6D(id, wrk_memlo, wrk_memhi)

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'EF_Util.parm'
      INCLUDE 'EF_mem_subsc.cmn'

* argument declarations
      INTEGER id, wrk_memlo(nferdims,EF_MAX_WORK_ARRAYS),
     .            wrk_memhi(nferdims,EF_MAX_WORK_ARRAYS)

* this assumed EF_MAX_WORK_ARRAYS .EQ. 9
      wrk_memlo(1,1) = wrk1lox
      wrk_memlo(1,2) = wrk2lox
      wrk_memlo(1,3) = wrk3lox
      wrk_memlo(1,4) = wrk4lox
      wrk_memlo(1,5) = wrk5lox
      wrk_memlo(1,6) = wrk6lox
      wrk_memlo(1,7) = wrk7lox
      wrk_memlo(1,8) = wrk8lox
      wrk_memlo(1,9) = wrk9lox

      wrk_memlo(2,1) = wrk1loy
      wrk_memlo(2,2) = wrk2loy
      wrk_memlo(2,3) = wrk3loy
      wrk_memlo(2,4) = wrk4loy
      wrk_memlo(2,5) = wrk5loy
      wrk_memlo(2,6) = wrk6loy
      wrk_memlo(2,7) = wrk7loy
      wrk_memlo(2,8) = wrk8loy
      wrk_memlo(2,9) = wrk9loy

      wrk_memlo(3,1) = wrk1loz
      wrk_memlo(3,2) = wrk2loz
      wrk_memlo(3,3) = wrk3loz
      wrk_memlo(3,4) = wrk4loz
      wrk_memlo(3,5) = wrk5loz
      wrk_memlo(3,6) = wrk6loz
      wrk_memlo(3,7) = wrk7loz
      wrk_memlo(3,8) = wrk8loz
      wrk_memlo(3,9) = wrk9loz

      wrk_memlo(4,1) = wrk1lot
      wrk_memlo(4,2) = wrk2lot
      wrk_memlo(4,3) = wrk3lot
      wrk_memlo(4,4) = wrk4lot
      wrk_memlo(4,5) = wrk5lot
      wrk_memlo(4,6) = wrk6lot
      wrk_memlo(4,7) = wrk7lot
      wrk_memlo(4,8) = wrk8lot
      wrk_memlo(4,9) = wrk9lot

      wrk_memlo(5,1) = wrk1loe
      wrk_memlo(5,2) = wrk2loe
      wrk_memlo(5,3) = wrk3loe
      wrk_memlo(5,4) = wrk4loe
      wrk_memlo(5,5) = wrk5loe
      wrk_memlo(5,6) = wrk6loe
      wrk_memlo(5,7) = wrk7loe
      wrk_memlo(5,8) = wrk8loe
      wrk_memlo(5,9) = wrk9loe

      wrk_memlo(6,1) = wrk1lof
      wrk_memlo(6,2) = wrk2lof
      wrk_memlo(6,3) = wrk3lof
      wrk_memlo(6,4) = wrk4lof
      wrk_memlo(6,5) = wrk5lof
      wrk_memlo(6,6) = wrk6lof
      wrk_memlo(6,7) = wrk7lof
      wrk_memlo(6,8) = wrk8lof
      wrk_memlo(6,9) = wrk9lof

      wrk_memhi(1,1) = wrk1hix
      wrk_memhi(1,2) = wrk2hix
      wrk_memhi(1,3) = wrk3hix
      wrk_memhi(1,4) = wrk4hix
      wrk_memhi(1,5) = wrk5hix
      wrk_memhi(1,6) = wrk6hix
      wrk_memhi(1,7) = wrk7hix
      wrk_memhi(1,8) = wrk8hix
      wrk_memhi(1,9) = wrk9hix

      wrk_memhi(2,1) = wrk1hiy
      wrk_memhi(2,2) = wrk2hiy
      wrk_memhi(2,3) = wrk3hiy
      wrk_memhi(2,4) = wrk4hiy
      wrk_memhi(2,5) = wrk5hiy
      wrk_memhi(2,6) = wrk6hiy
      wrk_memhi(2,7) = wrk7hiy
      wrk_memhi(2,8) = wrk8hiy
      wrk_memhi(2,9) = wrk9hiy

      wrk_memhi(3,1) = wrk1hiz
      wrk_memhi(3,2) = wrk2hiz
      wrk_memhi(3,3) = wrk3hiz
      wrk_memhi(3,4) = wrk4hiz
      wrk_memhi(3,5) = wrk5hiz
      wrk_memhi(3,6) = wrk6hiz
      wrk_memhi(3,7) = wrk7hiz
      wrk_memhi(3,8) = wrk8hiz
      wrk_memhi(3,9) = wrk9hiz

      wrk_memhi(4,1) = wrk1hit
      wrk_memhi(4,2) = wrk2hit
      wrk_memhi(4,3) = wrk3hit
      wrk_memhi(4,4) = wrk4hit
      wrk_memhi(4,5) = wrk5hit
      wrk_memhi(4,6) = wrk6hit
      wrk_memhi(4,7) = wrk7hit
      wrk_memhi(4,8) = wrk8hit
      wrk_memhi(4,9) = wrk9hit

      wrk_memhi(5,1) = wrk1hie
      wrk_memhi(5,2) = wrk2hie
      wrk_memhi(5,3) = wrk3hie
      wrk_memhi(5,4) = wrk4hie
      wrk_memhi(5,5) = wrk5hie
      wrk_memhi(5,6) = wrk6hie
      wrk_memhi(5,7) = wrk7hie
      wrk_memhi(5,8) = wrk8hie
      wrk_memhi(5,9) = wrk9hie

      wrk_memhi(6,1) = wrk1hif
      wrk_memhi(6,2) = wrk2hif
      wrk_memhi(6,3) = wrk3hif
      wrk_memhi(6,4) = wrk4hif
      wrk_memhi(6,5) = wrk5hif
      wrk_memhi(6,6) = wrk6hif
      wrk_memhi(6,7) = wrk7hif
      wrk_memhi(6,8) = wrk8hif
      wrk_memhi(6,9) = wrk9hif

      RETURN
      END

