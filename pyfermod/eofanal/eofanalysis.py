#! python
#

'''
Module defining the EOFAnalysis class for performing an 
empirical orthogonal function analysis of space-time data.  
Also defines the InvalidStateError exception that is 
raised by some methods in the EOFAnalysis class.

@author: Karl M. Smith
'''

from __future__ import print_function

import math
import numpy
import numpy.linalg



class InvalidStateError(Exception):
    '''
    Exception raised by methods of the EOFAnalysis class when
    an instance is not in a proper state for that function call.
    For example, results-reporting methods raise this error if
    called prior to calling the analyze method.
    '''
    pass



class EOFAnalysis(object):
    '''
    Class for performing an Empirical Orthogonal Function (EOF) analysis 
    of space-time data.  Create an instance of this class with the desired 
    space-time data array.  Call the analyze method of this instance to 
    perform the analysis.  Result-reporting methods can then be called to 
    obtain the results the of analysis.  The separate initialization, 
    analysis, and result-reporting methods allow for the creation of 
    methods to modify the input data prior to analysis, and methods to 
    modify the results prior to reporting.
    '''


    def __init__(self, spacetimedata):
        '''
        Initializes an EOFAnalysis instance using the given space-time data.

        Arguments:
            spacetimedata - matrix-like object of data where each column 
                    is a time series at a given location.  Thus the shape 
                    of the (row-major) data array should be (NT, NL) where 
                    where NT is the number of time values at each location 
                    and NL is the number of locations.

        Returns:
            an EOFAnalysis instance using the given space-time data.

        Raises:
            TypeError of ValueError - if spacetimedata cannot be made into
                    a (2D) numpy.matrix of numeric values.  See: numpy.matrix
            UserWarning - if there is one one time value in each time 
                    series.
        '''
        # matrix of averages of each time series
        self.__tavg = None
        # F matrix in EOF analysis = (origvals - tavg)
        self.__fmat = None
        # R matrix = F.T * F / NT (matrix multiplication)
        # (matrix of covariance means) - not saved
        # eigenvalues of R ordered from smallest to largest
        self.__eigvals = None
        # eigenvectors in columns of R corresponding to the eigenvalues
        self.__eigvecs = None
        # fraction significance of the corresponding EOFs
        self.__fracsignifs = None
        # minimum fraction ever considered significant; 0.01 == 1%
        self.__minsignif = 0.01
        # copy of the original space-time values as an matrix
        self.__origvals = numpy.matrix(spacetimedata, dtype=numpy.float64);
        if self.__origvals.shape[0] == 1:
            raise UserWarning('Only one time value in each time series')


    def analyze(self):
        '''
        Perform the empirical orthogonal function analysis of the
        space-time data contained in this EOFAnalysis instance.

        Arguments:
            None

        Returns:
            None

        Raises:
            eofanlysis.InvalidStateError - if the instance has not been 
                    properly initialized with space-time data.
            numpy.linlag.LinAlgError - if there is a problem with the 
                    analysis (diagonalizing the matrix of covariance means).
            UserWarning - if there is no variation found in the time series
     		    (no significant eigenvalues)
        '''
        if self.__origvals == None:
            raise InvalidStateError(
                    'instance data has not been properly initialized')
        # Create a matrix with each row containing the mean value
        # of each time-series in the original data.
        self.__tavg = numpy.average(self.__origvals, axis=0)
        # F is the data adjusted so each time series has a zero mean value.
        # self.__tavg is automatically repeated in the following 
        self.__fmat = self.__origvals - self.__tavg
        # R is the matrix of mean covariances over time = F.T * F / NT
        # R has shape (NP, NP)
        # Since fmat is a matrix, this is performing matrix multiplication.
        rmat = self.__fmat.T * self.__fmat / float(self.__fmat.shape[0])
        # Compute the eigenvalues and eigenvactors of the R matrix.
        # The numpy.linalg.eigh function returns eigenvalues from smallest 
        # to largest, so the significant EOFs are the last eigenvectors.
        # The eigenvectors are in the columns of the eigvectors matrix.
        (eigvals, eigvecs) = numpy.linalg.eigh(rmat)
        # Just making sure eigvals is an array and eigvecs is a matrix.
        self.__eigvals = numpy.array(eigvals, copy=False)
        self.__eigvecs = numpy.matrix(eigvecs, copy=False)
        # Compute the fraction significance of each EOF, which is the 
        # fraction each eigenvalue adds to the sum of all the eigenvalues.
        # Because R = R.T, all eigenvalues are non-negative or negligible
        # real values.
        eigvalsum = numpy.sum(self.__eigvals)
        if eigvalsum >= 1.0E-10:
            self.__fracsignifs = self.__eigvals / numpy.sum(self.__eigvals)
        else:
            # No significant eigenvalues == no significant EOFs
            self.__fracsignifs = numpy.zeros(self.__eigvals.shape)
            raise UserWarning(
     		'No variation in time series (no significant eigenvalues)')


    def signiffracs(self):
        '''
        Returns the significances of each of the empirical orthogonal
        functions (EOFs).  These values are the fraction of the total
        variability explained by each of the EOFs.

        Arguments:
            None

        Returns:
            an array of length NL, the number of locations (and thus,
            the maximum number of EOFs possible), giving the significance
            of the corresponding EOF.  Ordered from most significant to
            least significant.

        Raises:
            eofanalysis.InvalidStateError - if the analyze method has not 
                    been called.  See: EOFAnalysis.analyze
        '''
        if self.__fracsignifs == None:
            raise InvalidStateError('analyze method has not been called')
        # Return a copy of the computed significance of each EOF
        # ordered from largest to smallest (reversed).
        return numpy.array(self.__fracsignifs[::-1])


    def minsignif(self):
        '''
        Returns the current minimum fractional significance value used for
        determining which empirical orthogonal functions are significant.

        Arguments:
            None

        Returns:
            the current minimum fractions significance value
        '''
        return self.__minsignif


    def setminsignif(self, minsignif):
        '''
        Resets the minimum fractional significance value used for
        determining which empirical orthogonal functions are significant.

        Arguments:
            minsignif - value in the range [1.0E-6, 1.0 - 1.0E-6].
                    The 1.0E-6 (0.0001%) restriction is to circumvent
                    issues with numerical accuracy in the calculating
                    eigenvalues.

        Returns:
            None

        Raises:
            ValueError - if the value cannot be interpreted as a
                    numerical value, if the value is less than 1.0E-6,
                    or if the value is greater than (1.0 - 1.0E-6).
        '''
        newsignif = float(minsignif)
        if newsignif < 1.0E-6:
            raise ValueError('minsignif is less than 1.0E-6')
        if newsignif > 1.0 - 1.0E-6:
            raise ValueError('minsignif is greater than (1.0 - 1.0E-6)')
        self.__minsignif = newsignif


    def numeofs(self):
        '''
        Returns the number of significant empirical orthogonal functions
        (EOFs).  These are EOFs with a fraction significance not less
        than the current minimum significance value, which by default is
        0.01 (1%).  See: EOFAnalysis.minsignif and EOFAnalysis.setminsignif

        Arguments:
            None

        Returns:
            the number of significant EOFs.

        Raises:
            eofanalysis.InvalidStateError - if the analyze method has not 
                    been called.  See: EOFAnalysis.analyze
        '''
        if self.__fracsignifs == None:
            raise InvalidStateError('analyze method has not been called')
        # The fracsignifs are ordered from smallest to largest.
        # Optimized for the usual case where most EOFs are insignificant.
        for k in range(len(self.__fracsignifs)):
            if self.__fracsignifs[-(k+1)] < self.__minsignif:
                return k
        return len(self.__fracsignifs)


    def eofvec(self, num):
        '''
        Returns an empirical orthogonal function (EOF) as an array of
        location values in units of the original data.  Different EOFs
        are orthogonal to each other.  The square of the norm of an
        EOF is the eigenvalue associated with that EOF.

        Arguments:
            num -   positive integer giving the number of the EOF to
                    return.  A value of one gives the most influential
                    EOF, two gives the second-most influential EOF, etc.

        Returns:
            the requested EOF as a 1-D array of location values
            in units of the original data.

        Raises:
            ValueError - if num is not a positive integer or if num
                    is larger than the number of significant EOFs.
                    See: EOFAnalysis.numeofs
            eofanalysis.InvalidStateError - if the analyze method has 
                    not been called.  See: EOFAnalysis.analyze
        '''
        if self.__eigvals == None:
            raise InvalidStateError('analyze method has not been called')
        eofnum = int(num)
        if eofnum <= 0:
            raise ValueError('num is not a positive integer')
        if eofnum > len(self.__eigvals):
            raise ValueError('num is larger than the number of EOFs')
        # Eigenvalues are order from smallest to largest; thus, the
        # most significant EOF is derived from the last eigenvector.
        eofnum *= -1
        if self.__fracsignifs[eofnum] < self.__minsignif:
            raise ValueError(
                    'num is larger than the number of significant EOFs')
        # Eigenvectors are in the columns of the eigenvector matrix.
        # Multiply by the square root of the eigenvector to convert 
        # to units of the original data.
        eofvector = numpy.array(self.__eigvecs[:,eofnum]).squeeze() \
                    * math.sqrt(self.__eigvals[eofnum])
        return eofvector


    def tafvec(self, num):
        '''
        Returns the time-amplitude function (TAF) associated with an
        empirical orthogonal function (EOF) as a unitless array.  TAFs
        are orthogonal to each other.  The square of the norm of a TAF 
        equals the number of time values in each time series.

        Arguments:
            num -   positive integer giving the number of the TAF to
                    return.  A value of one gives the TAF associated
                    with the most influential EOF, two gives the TAF
                    associated with the second-most influential EOF, etc.

        Returns:
            the requested TAF as a 1-D array of unitless values.  The
            length of the array is NT, the number of time values in each
            time series.

        Raises:
            ValueError - if num is not a positive integer or is larger 
                    than the number of significant EOFs.
                    See: EOFAnalysis.numeofs
            eofanalysis.InvalidStateError - if the analyze method has not 
                    been called.  See: EOFAnalysis.analyze
        '''
        if self.__eigvals == None:
            raise InvalidStateError('analyze method has not been called')
        eofnum = int(num)
        if eofnum <= 0:
            raise ValueError('num is not a positive integer')
        if eofnum > len(self.__eigvals):
            raise ValueError('num is larger than the number of EOFs')
        # Eigenvalues are order from smallest to largest; thus, the
        # most significant EOF is derived from the last eigenvector.
        eofnum *= -1
        if self.__fracsignifs[eofnum] < self.__minsignif:
            raise ValueError(
                    'num is larger than the number of significant EOFs')
        # Time series are in the columns of the F matrix and eigenvectors are
        # in the columns of the eigenvector matrix.  Matrix multiplication
        # is being performed, and then the product is turned into a 1-D array.
        # Divide by the square root of the eigenvalue to  partially normalize
        # the vector.
        tafvector = numpy.array(self.__fmat * self.__eigvecs[:,eofnum]) \
                         .squeeze() / math.sqrt(self.__eigvals[eofnum])
        return tafvector


    def nullvec(self, num):
        '''
        Returns a vector of the null-space in this empirical orthogonal 
        function (EOF) analysis.  These are the eigenvectors corresponding 
        to EOFs with negligible significance.  Null-space vectors have a 
        norm of one and are orthogonal to each other.

        Since null-space vectors are orthogonal to the data, there are no 
        time amplitude functions (TAFs) associated with these vectors.
        (Or, if you prefer, the TAF is an array of zeros.)

        Arguments:
            num -   positive integer giving the number of the null-space 
                    vector to return.  There is no hierarchy for ordering 
                    null-space vectors.  The number of null-space vectors 
                    present is the number of locations less the number of
                    significant EOFs.  See: EOFAnalysis.numeofs

        Returns:
            the requested null-space vector as a 1-D array of location values.

        Raises:
            ValueError - if num is not a positive integer or is larger
                    than the number of null-space vectors.
            eofanalysis.InvalidStateError - if the analyze method has not 
                    been called.  See: EOFAnalysis.analyze
        '''
        if self.__eigvals == None:
            raise InvalidStateError('analyze method has not been called')
        nullnum = int(num)
        if nullnum <= 0:
            raise ValueError('num is not a positive integer')
        if nullnum > len(self.__eigvals):
            raise ValueError('num is larger than the number of locations')
        nullnum -= 1
        if self.__fracsignifs[nullnum] >= self.__minsignif:
            raise ValueError(
                    'num is larger than the number of null-space vectors')
        nullvector = numpy.array(self.__eigvecs[:,nullnum]).squeeze()
        return nullvector


    def datapiece(self, num):
        '''
        Returns a space-time matrix of computed data that is the portion of
        the original data explained by the indicated empirical orthogonal
        function (EOF) and corresponding time-amplitude function (TAF).

        When the argument is zero, a matrix of repeated time-series
        averages is returned.

        When the argument is positive, the value returned will be the 
        same as TAFmat * EOFmat.T, where TAFmat is a column matrix of
        the TAF values and EOFmat.T is a row matrix of the EOF values.
        However, a simplified formula is actually used in this method.  

        Summing the returned matrices for all significant EOFs will produce 
        a space-time matrix with negligible difference from the original data.  
        However, see EOFAnalysis.dataexplaned for a direct method for 
        obtaining these sums.

        Arguments:
            num -   non-negative integer.  If zero, returns the matrix of
                    repeated time-series averages; if positive, specifies
                    the EOF and TAF to use for computing the contribution.
                    A value of one computes the most significant contribution,
                    two gives the second-most significant contribution, etc.

        Returns:
            the computed space-time data contribution as designated by
            the argument.

        Raises:
            ValueError - if num is not a non-negative integer or is larger
                    than the number of EOFs.  See: EOFAnalysis.numeofs
            eofanalysis.InvalidStateError - if the analyze method has not 
                    been called.  See: EOFAnalysis.analyze
        '''
        if self.__eigvals == None:
            raise InvalidStateError('analyze method has not been called')
        eofnum = int(num)
        if eofnum == 0:
            # return a copy of the matrix with repeated time-series averages
            return self.__tavg.repeat(self.__origvals.shape[0], axis = 0)
        if eofnum < 0:
            raise ValueError('num is not a non-negative integer')
        if eofnum > len(self.__eigvals):
            raise ValueError('num is larger than the number of EOFs')
        # Eigenvalues are order from smallest to largest; thus, the
        # most significant EOF is derived from the last eigenvector.
        eofnum *= -1
        # Get the eofnum most significant eigenvectors as a matrix
        # vect shape = (np, 1)
        vect = self.__eigvecs[:,eofnum]
        # Project fmat - shape (nt, np) - into the subspace defined by this
        # eigenvector.  Using matrix multiplication; amat shape (nt, 1)
        amat = self.__fmat * vect
        # Bring this projection back into the complete space.
        # Matrix multiplication; vect.T shape (1, np); datapart shape (nt, np)
        datapart = amat * vect.T
        return datapart


    def dataexplained(self, num):
        '''
        Returns a space-time matrix of computed data that is the portion
        of the original data explained by the indicated number of the
        most-influential empirical orthogonal functions (EOFs) and their
        corresponding time amplitude functions (TAFs).

        This will be the same as the sum of the return values from
        EOFAnalysis.datapiece called with arguments from zero to num.
        However, a simplified formula is actually used in this method.  

        Arguments:
            num -   non-negative integer.  If zero, returns the matrix of
                    repeated time-series averages; if positive, specifies
                    the number of most-influential EOFs and TAFs to use in 
                    creating the computed data.  A value of one uses only 
                    the most influential EOF and TAF, two uses the two 
                    most-influential EOFs and TAFs, etc.

        Returns:
            the computed space-time data that can be directly compared
            to the original location-time data.

        Raises:
            ValueError - if num is not a non-negative integer or is larger 
                    than the number of EOFs.  See: EOFAnalysis.numeofs
            eofanalysis.InvalidStateError - if the analyze method has not 
                    been called.  See: EOFAnalysis.analyze
        '''
        if self.__eigvals == None:
            raise InvalidStateError('analyze method has not been called')
        eofnum = int(num)
        if eofnum == 0:
            # return a copy of the matrix with repeated time-series averages
            return self.__tavg.repeat(self.__origvals.shape[0], axis = 0)
        if eofnum < 0:
            raise ValueError('num is not a positive integer')
        if eofnum > len(self.__eigvals):
            raise ValueError('num is larger than the number of EOFs')
        # Eigenvalues are order from smallest to largest; thus, the
        # most significant EOF is derived from the last eigenvector.
        eofnum *= -1
        # Get the eofnum most significant eigenvectors as a matrix;
        # shape = (np, num)
        vecmat = self.__eigvecs[:,-1:eofnum-1:-1]
        # Project fmat - shape (nt, np) - into the subspace defined by these
        # eigenvectors.  Using matrix multiplication; amat shape = (nt, num)
        amat = self.__fmat * vecmat
        # Bring this projection back into the complete space and add the
        # time averages to make the result comparable to the original data
        # Matrix multiplication; vecmat.T shape (num, np) and expdata shape
        # (nt, np)
        expdata = (amat * vecmat.T) + self.__tavg
        return expdata


#
# The following is just for test "by-hand" and to serve as some examples.
# See test_eofanalysis.py for tests using the unittest module.
#

if __name__ == '__main__':
    import pprint
    formatter = pprint.PrettyPrinter()

    # five years of quarter-monthly data
    months = numpy.arange(0.0, 60.0, 0.25)

    # create time series on an annual cycle
    cosT = numpy.cos(months * numpy.pi / 6.0)
    sinT = numpy.sin(months * numpy.pi / 6.0)

    print()
    print('spacetime = [ cosT, sinT + 1 ]')
    spacetimedata = numpy.matrix([cosT,
                                  sinT + 1.0]).T
    eofanal = EOFAnalysis(spacetimedata)
    eofanal.analyze()
    defminsignif = eofanal.minsignif();
    eofanal.setminsignif(0.1)
    fracsignifs = eofanal.signiffracs()
    print('EOF fractional significances:')
    print(formatter.pformat(fracsignifs))
    numeofs = eofanal.numeofs()
    print('Number of significant EOFs: %d' % numeofs)
    totalcontrib = eofanal.datapiece(0)
    for k in range(1, numeofs+1):
        eofvec = eofanal.eofvec(k)
        sqnorm = numpy.dot(eofvec, eofvec)
        print('EOF %d has norm^2: %#.4f' % (k, sqnorm))
        print(formatter.pformat(eofvec))
        tafvec = eofanal.tafvec(k)
        sqnorm = numpy.dot(tafvec, tafvec)
        print('TAF %d has norm^2: %#.4f' % (k, sqnorm))
        print(formatter.pformat(tafvec))
        tafeof = numpy.outer(tafvec, eofvec) 
        contrib = eofanal.datapiece(k)
        if numpy.allclose(contrib, tafeof):
            print('datapiece(%d) all close to expected values' % k)
        else:
            raise ValueError(
                    'datapiece(%d):\n    expected\n%s\n    found\n%s' % \
                    (k, formatter.pformat(tafeof),
                        formatter.pformat(contrib)))
        totalcontrib += contrib
        expdata = eofanal.dataexplained(k)
        if numpy.allclose(expdata, totalcontrib):
            print('dataexplained(%d) all close to expected values' % k)
        else:
            raise ValueError(
                    'dataexplained(%d):\n    expected\n%s\n    found\n%s' % \
                    (k, formatter.pformat(totalcontrib),
                        formatter.pformat(expdata)))
        datadeltas = numpy.array(numpy.abs(spacetimedata - expdata)).flatten()
        maxdiff = numpy.max(datadeltas)
        rmsdiff = math.sqrt(numpy.average(datadeltas * datadeltas))
        print('Max and RMS diff btwn data explained by %d most' % k)
        print('    influential EOFs and actual data: %#.8f, %#.8f' % \
              (maxdiff, rmsdiff))
    datadeltas = numpy.array(numpy.abs(spacetimedata - totalcontrib)).flatten()
    maxdiff = numpy.max(datadeltas)
    rmsdiff = math.sqrt(numpy.average(datadeltas * datadeltas))
    print('Max and RMS diff btwn sum of all significant')
    print('    data pieces and actual data: %#.8f, %#.8f' % \
          (maxdiff, rmsdiff))

    print()
    print('spacetime = [ cosT * cosT, cosT * sinT + 1, cosT * sinT + 2, sinT * sinT + 3 ]')
    spacetimedata = numpy.matrix([ cosT * cosT,
                                   cosT * sinT + 1.0,
                                   cosT * sinT + 2.0,
                                   sinT * sinT + 3.0 ]).T
    eofanal = EOFAnalysis(spacetimedata)
    eofanal.analyze()
    fracsignifs = eofanal.signiffracs()
    print('EOF fractional significances:')
    print(formatter.pformat(fracsignifs))
    numeofs = eofanal.numeofs()
    print('Number of significant EOFs: %d' % numeofs)
    totalcontrib = eofanal.datapiece(0)
    for k in range(1, numeofs+1):
        eofvec = eofanal.eofvec(k)
        sqnorm = numpy.dot(eofvec, eofvec)
        print('EOF %d has norm^2: %#.4f' % (k, sqnorm))
        print(formatter.pformat(eofvec))
        tafvec = eofanal.tafvec(k)
        sqnorm = numpy.dot(tafvec, tafvec)
        print('TAF %d has norm^2: %#.4f' % (k, sqnorm))
        print(formatter.pformat(tafvec))
        tafeof = numpy.outer(tafvec, eofvec) 
        contrib = eofanal.datapiece(k)
        if numpy.allclose(contrib, tafeof):
            print('datapiece(%d) all close to expected values' % k)
        else:
            raise ValueError(
                    'datapiece(%d):\n    expected\n%s\n    found\n%s' % \
                    (k, formatter.pformat(tafeof),
                        formatter.pformat(contrib)))
        totalcontrib += contrib
        expdata = eofanal.dataexplained(k)
        if numpy.allclose(expdata, totalcontrib):
            print('dataexplained(%d) all close to expected values' % k)
        else:
            raise ValueError(
                    'dataexplained(%d):\n    expected\n%s\n    found\n%s' % \
                    (k, formatter.pformat(totalcontrib),
                        formatter.pformat(expdata)))
        datadeltas = numpy.array(numpy.abs(spacetimedata - expdata)).flatten()
        maxdiff = numpy.max(datadeltas)
        rmsdiff = math.sqrt(numpy.average(datadeltas * datadeltas))
        print('Max and RMS diff btwn data explained by %d most' % k)
        print('    influential EOFs and actual data: %#.8f, %#.8f' % \
              (maxdiff, rmsdiff))
    datadeltas = numpy.array(numpy.abs(spacetimedata - totalcontrib)).flatten()
    maxdiff = numpy.max(datadeltas)
    rmsdiff = math.sqrt(numpy.average(datadeltas * datadeltas))
    print('Max and RMS diff btwn sum of all significant')
    print('    data pieces and actual data: %#.8f, %#.8f' % \
          (maxdiff, rmsdiff))

    fmat = spacetimedata - eofanal.datapiece(0)
    for k in range(1, spacetimedata.shape[1] - numeofs + 1):
        nullvec = eofanal.nullvec(k)
        sqnorm = numpy.dot(nullvec, nullvec)
        print('Null-space vector %d has norm^2: %#.4f' % (k, sqnorm))
        print(formatter.pformat(nullvec))
        tafvec = numpy.array(fmat * numpy.matrix(nullvec).T).squeeze()
        sqnorm = numpy.dot(tafvec, tafvec)
        print('F * NSV %d has norm^2: %#.4f' % (k, sqnorm))
        # print formatter.pformat(tafvec)

    import pickle

    eofpicklestring = pickle.dumps(eofanal)
    print('length of the eofanal pickle string: %d' % len(eofpicklestring))
    neweofanal = pickle.loads(eofpicklestring)
    print('unpickled eofanal.numeofs() = %d' % neweofanal.numeofs())

