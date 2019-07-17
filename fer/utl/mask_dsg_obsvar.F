	SUBROUTINE MASK_DSG_OBSVAR (nfeatures, process_feature, 
     .        process_obs, rowsize, dset, cx, source_data, put_data, 
     .        nmasked)
     
* 
*  This software was developed by the Science Data Integration Group (SDIG) 
*  of the National Oceanographic and AtmosphericAdministration's (NOAA) 
*  Pacific Marine Environmental Lab(PMEL) as part of Ferret/PyFerret
*  5/1/2019
*
* Apply observation-level masking for data representing one or more 
* features of a Discrete Sampling Geometries dataset.


* V75+ 6/19 *acm*  Return the total number obs after masking

	LOGICAL process_feature(*), process_obs(*)
	INTEGER nfeatures, dset, cx, nmasked 
	REAL	rowsize(*), source_data(*), put_data(*)

	INTEGER base, fobs, ifeature, nobsf, iobs

	nmasked = 0
	base = 0     ! obs index at end of preceding feature
	fobs = 0     
	DO ifeature = 1, nfeatures
	   nobsf = rowsize(ifeature)  ! feature length
	   IF (process_feature(ifeature)) THEN

* ... apply observation-level mask for this feature
	      CALL MAKE_DSG_OBS_MASK(dset, cx, ifeature, base, process_obs, nobsf)
	      DO iobs = 1, nobsf
		 fobs = fobs+1
	         IF (process_obs(iobs)) THEN
		    nmasked = nmasked+1
		    put_data(nmasked) = source_data(fobs)		    
	         ENDIF
	      ENDDO   
	   ELSE
	      fobs = fobs + nobsf
	   ENDIF
	   
	   base = base + nobsf
	ENDDO

	RETURN
	END
	