'''
Maintains a dictionary of classes which can be used to bind PyFerret
graphics calls to a particular graphics engine.

When a new Window is needed, the createWindow function in this module
is called.  This function creates an instance of the appropriate
bindings class, its createWindow method is called, and the bindings 
instance is returned.

Other methods of the bindings instance are called directly to perform
graphics operations on this Window.
'''

from abstractpyferretbindings import AbstractPyFerretBindings

__pyferret_bindings_classes = { }

def addPyFerretBindings(engine_name, bindings_class):
    '''
    Adds the given class to the dictionary of available bindings.

    Arguments:
        engine_name: a string identifying the bindings class
                and graphics engine
        bindings_class: a subclass of AbstractPyFerretBindings
                providing the bindings for a graphics engine

    Raises:
        ValueError if bindings for engine_name already exist
        TypeError if bindings_class is not a subclass of
                AbstractPyFerretBindings
    '''
    try:
        _ = __pyferret_bindings_classes[engine_name]
        raise ValueError("Bindings already exist for graphics engine '%s'" \
                         % engine_name)
    except KeyError:
        pass
    if not issubclass(bindings_class, AbstractPyFerretBindings):
        raise TypeError("bindings_class argument is not a " \
                        "subclass of AbstractPyFerretBindings")
    __pyferret_bindings_classes[engine_name] = bindings_class

def knownPyFerretEngines():
    '''
    Returns a tuple of all the known engine names.
    '''
    return tuple( __pyferret_bindings_classes.keys() )
    
def createWindow(engine_name, title, visible, noalpha):
    '''
    Creates an instance of the bindings class associated with
    engine_name and calls the createWindow method of that
    instance of the class.  The instance of the bindings class
    is then returned.

    Arguments:
        engine_name: string identifying the bindings class and
                     graphics engine to use for the Window
        title: display title for the Window
        visible: display Window on start-up?
        noalpha: do not use the alpha channel in colors?

    Returns the instance of the binding class associated with
    the newly created Window if the createWindow method of the
    instance returns True.  Otherwise the bindings instance is
    deleted and None is returned.

    Raises a ValueError if engine_name does not have any bindings.
    '''
    try:
        bindclass = __pyferret_bindings_classes[engine_name]
    except KeyError:
        raise ValueError("Unknown graphics engine '%s'" % engine_name)
    bindinst = bindclass()
    if not bindinst.createWindow(title, visible, noalpha):
        del bindinst
        return None
    return bindinst


if __name__ == "__main__":

    class TestBindings(AbstractPyFerretBindings):
        engine_name = "TestEngine"

        def __init__(self):
            super(TestBindings, self).__init__()

        def createWindow(self, title, visible, noalpha):
            return True

    addPyFerretBindings(TestBindings.engine_name, TestBindings)
    bindinst = createWindow(TestBindings.engine_name, "test", False, False)
    if not bindinst:
        raise RuntimeError("Unsuccessful creation of a Window")
    try:
        addPyFerretBindings(TestBindings.engine_name, TestBindings)
        raise RuntimeError("Adding bindings with the same name succeeded")
    except ValueError:
        pass
    try:
        addPyFerretBindings("GenericObject", object)
        raise RuntimeError("Adding object for a bindings class succeeded")
    except TypeError:
        pass
    known_engines = knownPyFerretEngines()
    if (len(known_engines) != 1) or \
       (known_engines[0] != TestBindings.engine_name):
        raise RuntimeError("Unexpected tuple of known engines: %s" % \
                           str(known_engines))
    print "Success"

