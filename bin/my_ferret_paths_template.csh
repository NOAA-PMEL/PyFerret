## my_ferret_paths_template.csh
##
## Template for setting up a personal FERRET environment
## for users working under the tcsh or csh shells.
##
## Copy this file to your own directory area and
## customize it to suit your personal directory layout.
## Then source it (as below) from your .login file
## AFTER you source the generic ferret_paths
##
## example:  (in your .login file)
## source /usr/local/bin/ferret_paths    (or wherever your system mgr. has put it)
## source $HOME/my_ferret_paths

## These are the environment variables you may wish to customize.
## They are currently set up on the assumption that all your FERRET
## work is done in the directory $HOME/ferret .

   setenv FER_GO "$FER_GO $HOME/ferret"

   setenv FER_DATA "$FER_DATA $HOME/ferret"

   setenv FER_DESCR "$FER_DESCR $HOME/ferret"

   setenv FER_GRIDS "$FER_GRIDS $HOME/ferret"

   setenv FER_MODEL_RUNS "$HOME/ferret/model_runs"

