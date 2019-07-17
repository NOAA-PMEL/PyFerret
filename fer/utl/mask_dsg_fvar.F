	SUBROUTINE MASK_DSG_FVAR (nfeatures, process_feature, dset, 
     .        source_data, put_data)

* Argument declarations
	LOGICAL process_feature(*)
	INTEGER nfeatures, dset, line_to_put
	REAL source_data(*), put_data(*)

* Local declarations
	INTEGER count, ifeature


	count = 0
	DO ifeature = 1, nfeatures
	   IF (process_feature(ifeature)) THEN
	      count = count + 1
	      put_data(count) = source_data(ifeature)
	   ENDIF
	ENDDO

	RETURN
	END
