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
* This subroutine needs to be called at the start of the compute method
* of any PyFerret external function written in Fortran in order to make
* an external copy of the (Py)Ferret internal common block FERRET_EF_MEM_SUBSC.
* Because the PyFerret module is shared-object library loaded privately
* by Python, the Ferret common blocks are not visible to these external
* functions.
*

      SUBROUTINE COPY_FERRET_EF_MEM_SUBSC

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

* local variables
      INTEGER res_memlo(6), res_memhi(6)
      INTEGER arg_memlo(6, EF_MAX_ARGS)
      INTEGER arg_memhi(6, EF_MAX_ARGS)
      INTEGER wrk_memlo(6, EF_MAX_WORK_ARRAYS)
      INTEGER wrk_memhi(6, EF_MAX_WORK_ARRAYS)

* get the result array dimensions from Ferret's local copy of the common block
      CALL ef_get_res_mem_subscripts_6d(0, res_memlo, res_memhi)

* copy the result array dimensions to this externally visible common block
      memreslox = res_memlo(1)
      memresloy = res_memlo(2) 
      memresloz = res_memlo(3) 
      memreslot = res_memlo(4) 
      memresloe = res_memlo(5) 
      memreslof = res_memlo(6) 

      memreshix = res_memhi(1) 
      memreshiy = res_memhi(2) 
      memreshiz = res_memhi(3) 
      memreshit = res_memhi(4) 
      memreshie = res_memhi(5) 
      memreshif = res_memhi(6) 

* get the argument arrays dimensions from Ferret's local copy of the common block
      CALL ef_get_arg_mem_subscripts_6d(0, arg_memlo, arg_memhi)

* copy the argument arrays dimensions to this externally visible common block
* this assumed EF_MAX_ARGS .EQ. 9
      mem1lox = arg_memlo(1,1) 
      mem2lox = arg_memlo(1,2) 
      mem3lox = arg_memlo(1,3) 
      mem4lox = arg_memlo(1,4) 
      mem5lox = arg_memlo(1,5) 
      mem6lox = arg_memlo(1,6) 
      mem7lox = arg_memlo(1,7) 
      mem8lox = arg_memlo(1,8) 
      mem9lox = arg_memlo(1,9) 

      mem1loy = arg_memlo(2,1) 
      mem2loy = arg_memlo(2,2) 
      mem3loy = arg_memlo(2,3) 
      mem4loy = arg_memlo(2,4) 
      mem5loy = arg_memlo(2,5) 
      mem6loy = arg_memlo(2,6) 
      mem7loy = arg_memlo(2,7) 
      mem8loy = arg_memlo(2,8) 
      mem9loy = arg_memlo(2,9) 

      mem1loz = arg_memlo(3,1) 
      mem2loz = arg_memlo(3,2) 
      mem3loz = arg_memlo(3,3) 
      mem4loz = arg_memlo(3,4) 
      mem5loz = arg_memlo(3,5) 
      mem6loz = arg_memlo(3,6) 
      mem7loz = arg_memlo(3,7) 
      mem8loz = arg_memlo(3,8) 
      mem9loz = arg_memlo(3,9) 

      mem1lot = arg_memlo(4,1) 
      mem2lot = arg_memlo(4,2) 
      mem3lot = arg_memlo(4,3) 
      mem4lot = arg_memlo(4,4) 
      mem5lot = arg_memlo(4,5) 
      mem6lot = arg_memlo(4,6) 
      mem7lot = arg_memlo(4,7) 
      mem8lot = arg_memlo(4,8) 
      mem9lot = arg_memlo(4,9) 

      mem1loe = arg_memlo(5,1) 
      mem2loe = arg_memlo(5,2) 
      mem3loe = arg_memlo(5,3) 
      mem4loe = arg_memlo(5,4) 
      mem5loe = arg_memlo(5,5) 
      mem6loe = arg_memlo(5,6) 
      mem7loe = arg_memlo(5,7) 
      mem8loe = arg_memlo(5,8) 
      mem9loe = arg_memlo(5,9) 

      mem1lof = arg_memlo(6,1) 
      mem2lof = arg_memlo(6,2) 
      mem3lof = arg_memlo(6,3) 
      mem4lof = arg_memlo(6,4) 
      mem5lof = arg_memlo(6,5) 
      mem6lof = arg_memlo(6,6) 
      mem7lof = arg_memlo(6,7) 
      mem8lof = arg_memlo(6,8) 
      mem9lof = arg_memlo(6,9) 

      mem1hix = arg_memhi(1,1) 
      mem2hix = arg_memhi(1,2) 
      mem3hix = arg_memhi(1,3) 
      mem4hix = arg_memhi(1,4) 
      mem5hix = arg_memhi(1,5) 
      mem6hix = arg_memhi(1,6) 
      mem7hix = arg_memhi(1,7) 
      mem8hix = arg_memhi(1,8) 
      mem9hix = arg_memhi(1,9) 

      mem1hiy = arg_memhi(2,1) 
      mem2hiy = arg_memhi(2,2) 
      mem3hiy = arg_memhi(2,3) 
      mem4hiy = arg_memhi(2,4) 
      mem5hiy = arg_memhi(2,5) 
      mem6hiy = arg_memhi(2,6) 
      mem7hiy = arg_memhi(2,7) 
      mem8hiy = arg_memhi(2,8) 
      mem9hiy = arg_memhi(2,9) 

      mem1hiz = arg_memhi(3,1) 
      mem2hiz = arg_memhi(3,2) 
      mem3hiz = arg_memhi(3,3) 
      mem4hiz = arg_memhi(3,4) 
      mem5hiz = arg_memhi(3,5) 
      mem6hiz = arg_memhi(3,6) 
      mem7hiz = arg_memhi(3,7) 
      mem8hiz = arg_memhi(3,8) 
      mem9hiz = arg_memhi(3,9) 

      mem1hit = arg_memhi(4,1) 
      mem2hit = arg_memhi(4,2) 
      mem3hit = arg_memhi(4,3) 
      mem4hit = arg_memhi(4,4) 
      mem5hit = arg_memhi(4,5) 
      mem6hit = arg_memhi(4,6) 
      mem7hit = arg_memhi(4,7) 
      mem8hit = arg_memhi(4,8) 
      mem9hit = arg_memhi(4,9) 

      mem1hie = arg_memhi(5,1) 
      mem2hie = arg_memhi(5,2) 
      mem3hie = arg_memhi(5,3) 
      mem4hie = arg_memhi(5,4) 
      mem5hie = arg_memhi(5,5) 
      mem6hie = arg_memhi(5,6) 
      mem7hie = arg_memhi(5,7) 
      mem8hie = arg_memhi(5,8) 
      mem9hie = arg_memhi(5,9) 

      mem1hif = arg_memhi(6,1) 
      mem2hif = arg_memhi(6,2) 
      mem3hif = arg_memhi(6,3) 
      mem4hif = arg_memhi(6,4) 
      mem5hif = arg_memhi(6,5) 
      mem6hif = arg_memhi(6,6) 
      mem7hif = arg_memhi(6,7) 
      mem8hif = arg_memhi(6,8) 
      mem9hif = arg_memhi(6,9) 

* get the work arrays dimensions from Ferret's local copy of the common block
      CALL ef_get_wrk_mem_subscripts_6d(0, wrk_memlo, wrk_memhi)

* copy the work arrays dimensions to this externally visible common block
* this assumed EF_MAX_WORK_ARRAYS .EQ. 9
      wrk1lox = wrk_memlo(1,1) 
      wrk2lox = wrk_memlo(1,2) 
      wrk3lox = wrk_memlo(1,3) 
      wrk4lox = wrk_memlo(1,4) 
      wrk5lox = wrk_memlo(1,5) 
      wrk6lox = wrk_memlo(1,6) 
      wrk7lox = wrk_memlo(1,7) 
      wrk8lox = wrk_memlo(1,8) 
      wrk9lox = wrk_memlo(1,9) 

      wrk1loy = wrk_memlo(2,1) 
      wrk2loy = wrk_memlo(2,2) 
      wrk3loy = wrk_memlo(2,3) 
      wrk4loy = wrk_memlo(2,4) 
      wrk5loy = wrk_memlo(2,5) 
      wrk6loy = wrk_memlo(2,6) 
      wrk7loy = wrk_memlo(2,7) 
      wrk8loy = wrk_memlo(2,8) 
      wrk9loy = wrk_memlo(2,9) 

      wrk1loz = wrk_memlo(3,1) 
      wrk2loz = wrk_memlo(3,2) 
      wrk3loz = wrk_memlo(3,3) 
      wrk4loz = wrk_memlo(3,4) 
      wrk5loz = wrk_memlo(3,5) 
      wrk6loz = wrk_memlo(3,6) 
      wrk7loz = wrk_memlo(3,7) 
      wrk8loz = wrk_memlo(3,8) 
      wrk9loz = wrk_memlo(3,9) 

      wrk1lot = wrk_memlo(4,1) 
      wrk2lot = wrk_memlo(4,2) 
      wrk3lot = wrk_memlo(4,3) 
      wrk4lot = wrk_memlo(4,4) 
      wrk5lot = wrk_memlo(4,5) 
      wrk6lot = wrk_memlo(4,6) 
      wrk7lot = wrk_memlo(4,7) 
      wrk8lot = wrk_memlo(4,8) 
      wrk9lot = wrk_memlo(4,9) 

      wrk1loe = wrk_memlo(5,1) 
      wrk2loe = wrk_memlo(5,2) 
      wrk3loe = wrk_memlo(5,3) 
      wrk4loe = wrk_memlo(5,4) 
      wrk5loe = wrk_memlo(5,5) 
      wrk6loe = wrk_memlo(5,6) 
      wrk7loe = wrk_memlo(5,7) 
      wrk8loe = wrk_memlo(5,8) 
      wrk9loe = wrk_memlo(5,9) 

      wrk1lof = wrk_memlo(6,1) 
      wrk2lof = wrk_memlo(6,2) 
      wrk3lof = wrk_memlo(6,3) 
      wrk4lof = wrk_memlo(6,4) 
      wrk5lof = wrk_memlo(6,5) 
      wrk6lof = wrk_memlo(6,6) 
      wrk7lof = wrk_memlo(6,7) 
      wrk8lof = wrk_memlo(6,8) 
      wrk9lof = wrk_memlo(6,9) 

      wrk1hix = wrk_memhi(1,1) 
      wrk2hix = wrk_memhi(1,2) 
      wrk3hix = wrk_memhi(1,3) 
      wrk4hix = wrk_memhi(1,4) 
      wrk5hix = wrk_memhi(1,5) 
      wrk6hix = wrk_memhi(1,6) 
      wrk7hix = wrk_memhi(1,7) 
      wrk8hix = wrk_memhi(1,8) 
      wrk9hix = wrk_memhi(1,9) 

      wrk1hiy = wrk_memhi(2,1) 
      wrk2hiy = wrk_memhi(2,2) 
      wrk3hiy = wrk_memhi(2,3) 
      wrk4hiy = wrk_memhi(2,4) 
      wrk5hiy = wrk_memhi(2,5) 
      wrk6hiy = wrk_memhi(2,6) 
      wrk7hiy = wrk_memhi(2,7) 
      wrk8hiy = wrk_memhi(2,8) 
      wrk9hiy = wrk_memhi(2,9) 

      wrk1hiz = wrk_memhi(3,1) 
      wrk2hiz = wrk_memhi(3,2) 
      wrk3hiz = wrk_memhi(3,3) 
      wrk4hiz = wrk_memhi(3,4) 
      wrk5hiz = wrk_memhi(3,5) 
      wrk6hiz = wrk_memhi(3,6) 
      wrk7hiz = wrk_memhi(3,7) 
      wrk8hiz = wrk_memhi(3,8) 
      wrk9hiz = wrk_memhi(3,9) 

      wrk1hit = wrk_memhi(4,1) 
      wrk2hit = wrk_memhi(4,2) 
      wrk3hit = wrk_memhi(4,3) 
      wrk4hit = wrk_memhi(4,4) 
      wrk5hit = wrk_memhi(4,5) 
      wrk6hit = wrk_memhi(4,6) 
      wrk7hit = wrk_memhi(4,7) 
      wrk8hit = wrk_memhi(4,8) 
      wrk9hit = wrk_memhi(4,9) 

      wrk1hie = wrk_memhi(5,1) 
      wrk2hie = wrk_memhi(5,2) 
      wrk3hie = wrk_memhi(5,3) 
      wrk4hie = wrk_memhi(5,4) 
      wrk5hie = wrk_memhi(5,5) 
      wrk6hie = wrk_memhi(5,6) 
      wrk7hie = wrk_memhi(5,7) 
      wrk8hie = wrk_memhi(5,8) 
      wrk9hie = wrk_memhi(5,9) 

      wrk1hif = wrk_memhi(6,1) 
      wrk2hif = wrk_memhi(6,2) 
      wrk3hif = wrk_memhi(6,3) 
      wrk4hif = wrk_memhi(6,4) 
      wrk5hif = wrk_memhi(6,5) 
      wrk6hif = wrk_memhi(6,6) 
      wrk7hif = wrk_memhi(6,7) 
      wrk8hif = wrk_memhi(6,8) 
      wrk9hif = wrk_memhi(6,9) 

      RETURN
      END

