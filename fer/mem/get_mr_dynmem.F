	SUBROUTINE GET_MR_DYNMEM( i8_rqst_size, mr, status )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* allocate the requested amount of dynamic emory (using c or python malloc)
* malloc is done in c (or python), but has to call back into F90 to set pointer
* the hierarchy of calls is as follows
*   get_mr_dynmem.F ==> get_mr_mem.c ==> store_mr_ptr.F

* programmer - steve hankin
* V702 *sh*  1/17 trac enhancement ticket #2369 - dynamic memory
* v74   *acm* 3/18 Introduce parameter abstractax_len replacing 99999999

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'errmsg.parm'
	include	'ferret.parm'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'

* calling argument declarations:
	INTEGER	mr, status
	INTEGER*8 i8_rqst_size

* internal variable declarations:
	INTEGER	num_slen, mr_del
	INTEGER*8 available_mem
	CHARACTER  LEFINT8*20, TM_FMT*12, num_str*20

* Check for unspecified limits on abstract axis
	IF ( i8_rqst_size .EQ. abstractax_len ) GOTO 5100

* if we know there isn't room to accommodate this new object then
* we can error out without compromising the current contents of the cache
	available_mem = max_mem_allowed-(permanent_mem+essential_mem)
	IF ( i8_rqst_size .GT. available_mem) GOTO 5200

* will this new allocation push us over the specified memory size limit?
	IF (total_mem+i8_rqst_size .GT. max_mem_allowed) THEN
* ... delete from tail of deletion priority chain until there's room
 100	  mr_del = mr_del_blink(0)
	    IF ( mr_del .EQ. 0 ) THEN
* ... unable to clear out enough memory for this request
* ... should never get here because of pre-check above
	      CALL WARN("crptn??: get_mr_dynmem")
	      GOTO 5200   ! abort
	    ENDIF
	    CALL DELETE_VARIABLE( mr_del )   ! reduces total_mem
	  IF  (total_mem+i8_rqst_size .GT. max_mem_allowed) GOTO 100 !delete more 
	ENDIF

* allocate the dynamic memory
 200	CALL GET_MR_MEM(mr, i8_rqst_size, status)
        IF (status .NE. ferr_ok) GOTO 5300

* bookkeeping
        mr_size(mr) = i8_rqst_size
        total_mem = total_mem + i8_rqst_size
	IF (total_mem .GT. peak_mem) peak_mem = total_mem
	RETURN


***********************
* error exits
 5000	RETURN
 5100	num_str = LEFINT8(i8_rqst_size,num_slen)
	CALL ERRMSG( ferr_dim_underspec, status,
     .    num_str(:num_slen)//' words were requested. ', *5110)
 5110	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'    Check for unspecified limits on an ABSTRACT axis',0 )
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'    Use the SHOW GRID command to see the axes',0 )
	GOTO 5000

 5200	CALL ERRMSG( ferr_insuff_memory,status,' ', *5210)
 5210	num_str = TM_FMT( max_mem_allowed/1.E6, 6, 12, num_slen)
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'    To fulfill this request would exceed the current '//
     .		  'SET MEMORY/SIZE= limit of '//num_str(:num_slen)
     .	          //' megawords',0 )
	CALL SHOW_ABORTED_MEM_STATE(i8_rqst_size)
        GOTO 5000

 5300	CALL ERRMSG( ferr_insuff_memory,status,' ', *5310)
 5310	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'    The OS refuses to supply this much memory',0 )
	CALL SHOW_ABORTED_MEM_STATE(i8_rqst_size)
        GOTO 5000

	END


