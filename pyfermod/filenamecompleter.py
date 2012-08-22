#! python
#

"""
Module that defines a readline completer class with a complete 
method that returns filenames matching the given partial name.
Thus the readline suggestions from using this completer will 
be names of existing files that match the partially specified 
directory/file name.
"""

import os.path

class FilenameCompleter(object):
    """
    A readline completer class whose complete method returns 
    filenames matching the given partial name.  Thus the readline 
    suggestions from using this completer will be names of existing 
    files that match the partially specified directory/file name.
    """
    def __init__(self):
        self.__matches = []

    def complete(self, text, state):
        """
        Returns a filename matching the partial name given in text.

        If the partial name contains one single-quote character or 
        one double-quote character, everything after that quote 
        character is used as the beginning of the directory/file
        name to match (the "match string").  Otherwise, everything 
        after the last whitespace character is used as the match
        string.  In any case, any trailing whitespace characters 
        in the match string are removed.

        An initial component of ~/ or ~user/ in the match string
        is substituted with the home directory of the current or 
        specified user, respectively.
        See: os.path.expanduser

        Any environment variable names (such as ${HOME}) given in 
        the match string are substituted with the corresponding 
        value of the environment variable.
        See: os.path.expandvars

        If state is zero, a listing of the files matching the 
        match string is obtained, then sorted alphabetically, 
        and saved.  The first matching name is returned.

        If state is greater than zero, the name at that index 
        from the last saved listing is returned (and, thus, the
        value of text is ignored).  If state is larger than the 
        number of names in the last saved listing, None is returned.
        """
        if state == 0:
            # first call; get the substring to match
            if text.count("'") == 1:
                parttext = text[text.index("'") + 1:].rstrip()
            elif text.count('"') == 1:
                parttext = text[text.index('"') + 1:].rstrip()
            else:
                # use the last space-separated piece
                pieces = text.rsplit(None, 1)
                if len(pieces) == 0:
                   parttext = ''
                elif len(pieces) == 1:
                   parttext = pieces[0]
                else:
                   parttext = pieces[1]
            # expand any initial ~ or ~user
            exptext = os.path.expanduser(parttext)
            # expand any environment variables names
            exptext = os.path.expandvars(exptext)
            # split the match string into the directory path and (partial) filename
            (head, tail) = os.path.split(exptext)
            try:
                if head == '':
                    # use the contents of the current directory
                    dirlist = os.listdir(os.curdir)
                else:
                    # use the contents of the directory given by head
                    dirlist = os.listdir(head)
            except OSError:
                dirlist = []
            if tail == '':
                # use all the names given
                filteredlist = dirlist
            else:
                # match only those names beginning with the string in tail
                lentail = len(tail)
                filteredlist = [ fnam for fnam in dirlist 
                                 if fnam[:lentail] == tail ]
            # sort the filtered list of names
            filteredlist.sort()
            if head == '':
                # just use the names with no directory path
                self.__matches = filteredlist
            else:
                # prefix with the directory path
                self.__matches = [ os.path.join(head, fnam) 
                                   for fnam in filteredlist ]
        # Return the name at the index given by state
        try:
            return self.__matches[state]
        except IndexError:
            return None


#
#  The following is just for testing
#

if __name__ == '__main__':
    completer = FilenameCompleter()

    # Test an empty string
    actdirlist = os.listdir(os.curdir)
    actdirlist.sort()
    print 'Contents of current directory'
    cmpdirlist = []
    k = 0
    fnam = completer.complete('', 0)
    while fnam != None:
        print '    %s' % fnam
        cmpdirlist.append(fnam)
        k += 1
        fnam = completer.complete('', k)
    if cmpdirlist != actdirlist:
        raise ValueError('Empty text failure; expected: %s, found: %s' % \
                         (str(actdirlist), str(cmpdirlist)))

    # Test with a tilde string
    tildedir = '~' + os.sep
    print ''
    print 'Contents of %s' % tildedir
    tildenames = []
    k = 0
    fnam = completer.complete(tildedir, 0)
    while fnam != None:
        print '    %s' % fnam
        tildenames.append(fnam)
        k += 1
        fnam = completer.complete(tildedir, k)
    
    # Test with an environment variable
    homedir = '$HOME' + os.sep
    print ''
    print 'Contents of %s' % homedir
    homenames = []
    k = 0
    fnam = completer.complete(homedir, 0)
    while fnam != None:
        print '    %s' % fnam
        homenames.append(fnam)
        k += 1
        fnam = completer.complete(homedir, k)

    # ~ and $HOME should be the same
    if tildenames != homenames:
        raise ValueError('%s and %s lists do not match' % (tildedir, homedir))

    # Try with $HOME/bin/
    bindir = '$HOME' + os.sep + 'bin' + os.sep
    print ''
    print 'Contents of %s' % bindir
    binnames = []
    k = 0
    fnam = completer.complete(bindir, 0)
    while fnam != None:
        print '    %s' % fnam
        binnames.append(fnam)
        k += 1
        fnam = completer.complete(bindir, k)
    if binnames == homenames:
        raise ValueError('%s and %s lists match' % (bindir, homedir))

    # Try with an invalid directory
    invalid_name = 'hopefully' + os.sep + 'a' + os.sep + 'non' + os.sep \
                 + 'existant' + os.sep + 'directory' + os.sep + 'name'
    fnam = completer.complete(invalid_name, 0)
    if fnam != None:
        raise ValueError('complete "%s" failure; expected: None, found: %s' % \
                         (invalid_name, fnam))

    # Try with an unreadable directory (on unix systems)
    invalid_name = '/lost+found/'
    fnam = completer.complete(invalid_name, 0)
    if fnam != None:
        raise ValueError('complete "%s" failure; expected: None, found: %s' % \
                         (invalid_name, fnam))

    # All tests successful
    print ''
    print 'Success'

