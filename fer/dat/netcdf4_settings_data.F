	BLOCK DATA NETCDF4_LIST_SETTINGS_DATA
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
*  updates, revisions, new versions or 'bug fixes'.
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP 'AS IS' AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* Initialize the settings for netcdf4 datasets and store them in common.

        include 'netcdf4_settings.cmn'

*
*	netcdf4_type	- netcdf format: 4=netcdf4, 3=netcdf3(=classic), 6=64-bit
*	xchunk_size     - chunk size in X direction
*	ychunk_size     - chunk size in Y direction
*	zchunk_size     - chunk size in Z direction
*	tchunk_size     - chunk size in T direction
*	deflate_lev	- deflate level (0 for no deflate)
*	shuffle_flag    - shuffle - 0 or 1
*	endian_code     - native=0, little=1, big=0 match nf_endian_* params
*	cache_size_mb   - cache_chunk size in mbytes

	DATA netcdf4_type /3/
	DATA xchunk_size  /0/
	DATA ychunk_size  /0/
	DATA zchunk_size  /0/
	DATA tchunk_size  /0/
	DATA deflate_lev  /0/
	DATA shuffle_flag /0/
	DATA endian_code  /0/
	DATA netcdf4_type_set /3/
	DATA xchunk_size_set  /0/
	DATA ychunk_size_set  /0/
	DATA zchunk_size_set  /0/
	DATA tchunk_size_set  /0/
	DATA deflate_lev_set  /0/
	DATA shuffle_flag_set /0/
	DATA endian_code_set  /0/
	DATA default_cache_size  /0/

* cache_size_mb is obtained on initialization by a call to the netcdf library.

	END
