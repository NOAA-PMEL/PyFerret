	SUBROUTINE RESTORE_CURVI 
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
* Restore settings for dynamic-memory pointers for the curvilinear coordinate data
*
* programmer - Ansley Manke
* PMEL SDIG, 1/19/2018
*
* The curvilinear 2-D coordinate data is passed to PPLUS using the pointers to  
* memory. If the command is e.g. a SHADE/SET, then the data for the curvilinear 
* coordinates has been loaded but may be lost if the commands that come after 
* /SET and before PPL SHADE include redefining any variable, which triggers a 
* freeing of memory. The mr_protected flag has been set to prevent these 
* variables from being freed.  Now that the plot is done, restore the settings 
* for mr_protected.

	include 'tmap_dims.parm'
	include 'ferret.parm'
        include 'xvariables.cmn'
	include 'xcurvilinear.cmn'

	IF (xmem_ptr .NE. mr_nonexist) THEN
	   IF (mr_protected(xmem_ptr) .LT. mr_keep_flag) THEN
	      mr_protected(xmem_ptr) = mr_protected(xmem_ptr) - mr_keep_now
	      CALL DELETE_VARIABLE(xmem_ptr)
	   ENDIF
	ENDIF
	   IF (ymem_ptr .NE. mr_nonexist) THEN
	   IF (mr_protected(ymem_ptr) .LT. mr_keep_flag) THEN
	      mr_protected(ymem_ptr) = mr_protected(ymem_ptr) - mr_keep_now
	      CALL DELETE_VARIABLE(ymem_ptr)
	   ENDIF
	ENDIF

	RETURN
	END
