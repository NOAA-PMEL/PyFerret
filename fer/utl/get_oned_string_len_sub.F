
	INTEGER FUNCTION GET_ONED_STRING_LEN_SUB (grid_data, lo,hi, indx )
	
* Flattening out grid_data to treat it as a 1-D grid ignoring zero-length dimensions
* and return the length of the string at location indx

	INTEGER  indx, lo, hi
	REAL*8 grid_data( lo:hi )
	
	INTEGER GET_C_STRING_LEN

	GET_ONED_STRING_LEN_SUB = GET_C_STRING_LEN( grid_data(indx) )

	RETURN
	END
