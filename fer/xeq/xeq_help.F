	SUBROUTINE XEQ_HELP

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
* execute the HELP command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86 - dummy version
* revision 1.0 - 5/13/87 - calls VMS utility routines
* revision 1.1 - 3/7/88  - changed help file name to logical name
* revision 1.2 - 4/26/88 - added PUT_OUTPUT/GET_INPUT
* revision 1.3 - 6/22/88 - changed GFDL_HELP --> FERRET_HELP
* revision 1.4 - 8/3/88  - Top level of help library, "FERRET", removed
* V230: 8/26/92 - added Unix version - entirely separate from VMS
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V5.0: 1/1/99 - Start up appropriate Web browser on PC
* v500: 1/26/99 - *kob* update help to reflect current options

* V5.6  3/2004 - update help to point only to web resources and SHOW commands

        include 'ferret.parm'
        include 'xprog_state.cmn'


	CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .   ' Detailed help is available online at www.ferret.noaa.gov'
     .								, 0)

	CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .   ' To obtain a list of FERRET commands type    SHOW COMMANDS'
     .								, 0)

	CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .' To obtain a list of FERRET functions type    SHOW FUNCTIONS'
     .								, 0)

	CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .' To obtain a list of FERRET transforms type    SHOW TRANSFORM'
     .								, 0)

        RETURN

	END

