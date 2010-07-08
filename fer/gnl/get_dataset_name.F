        SUBROUTINE GET_DATASET_NAME_FROM_BROWSER(status)

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
* Get the name of a dataset for a SET DATA /BROWSE command
*
        INTEGER status

        include 'errmsg.parm'
        include 'ferret.parm'
        include 'xprog_state.cmn'

        INTEGER RUN_THREDDS_BROWSER, TM_LENSTR, len_name, buff_len
        CHARACTER*512 dataset_name
        CHARACTER*2048 err_warn_msg

* Run the browser to get the dataset name
        dataset_name = ' '
        err_warn_msg = ' '
        len_name = RUN_THREDDS_BROWSER(dataset_name, err_warn_msg)

* Check if there was an error running the browser
        IF ( len_name .LT. 0 ) THEN
            CALL SPLIT_LIST(pttmode_ops, jrnl_lun, 
     .                      '!-> **ERROR**', 0)
            CALL SPLIT_LIST(pttmode_ops, ttout_lun, 
     .                      '!-> **ERROR**', 0)
            CALL SPLIT_LIST(pttmode_ops, tterr_lun, err_warn_msg, 0)
            status = ferr_erreq
            RETURN
        ENDIF

* Check if the user canceled out
        IF ( len_name .EQ. 0 ) THEN
            CALL SPLIT_LIST(pttmode_ops, jrnl_lun, 
     .                      '!-> **CANCEL**', 0)
            CALL SPLIT_LIST(pttmode_ops, ttout_lun, 
     .                      '!-> **CANCEL**', 0)
            status = ferr_erreq
            RETURN
        ENDIF

* Append the dataset name obtained
        buff_len = TM_LENSTR(cmnd_buff)
        cmnd_buff(buff_len+1:) = ' "' // dataset_name(1:len_name) // '"'
        num_items = num_items + 1
        item_start(num_items) = buff_len + 3
        item_end(num_items) = item_start(num_items) + len_name - 1

* include the quotes in the filename printed to stdout and the journal
        CALL SPLIT_LIST(pttmode_ops, jrnl_lun, 
     .                  '!-> ' // cmnd_buff, 0)
        CALL SPLIT_LIST(pttmode_ops, ttout_lun, 
     .                  '!-> ' // cmnd_buff, 0)

        status = ferr_ok
        RETURN

        END
