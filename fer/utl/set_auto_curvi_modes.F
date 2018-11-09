	SUBROUTINE SET_AUTO_CURVI_MODES( auto_auxvar_changed )

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
* process /CURVILINEAR qualifiers from all commands

* *sh* 12/13
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'command.parm'
	include	'slash.parm'
	include 'xcommand.cmn'
	include 'xprog_state.cmn'

* internal variable declarations
	LOGICAL	auto_auxvar_changed


* TEMPORARY PLACE_HOLDER CODE.
*  will need to support /CURVI qualifiers on many commands

	mode_auto_curvi_last = mode_auto_curvi
	mode_auto_curvi  = mode_state(pmode_curvilinear,1)
     .		  .OR. (  (cmnd_num .EQ. cmnd_list)
     .	             .AND. qual_given(slash_list_curvi).GT.0 )

	mode_auto_layerz_last = mode_auto_layerz
	mode_auto_layerz = mode_state(pmode_layerz,1)
     .		  .OR. (  (cmnd_num .EQ. cmnd_list)
     .	             .AND. qual_given(slash_list_layerz).GT.0 )


	auto_auxvar_changed =
     .	     mode_auto_curvi  .NEQV. mode_auto_curvi_last
     . .OR.  mode_auto_layerz .NEQV. mode_auto_layerz_last


	RETURN
	END
