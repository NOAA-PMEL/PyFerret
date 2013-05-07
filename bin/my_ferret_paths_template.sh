## my_ferret_paths_template.sh
##
## Template for setting up a personal FERRET environment
## for users working under the bash, ksh, dash, or sh shells.
##
## Copy this file to your own directory area and
## customize it to suit your personal directory layout.
## Then source it (as below) from your .bashrc file
## AFTER you source the generic ferret_paths
##
## example:  (in your .bashrc file)
## . /usr/local/bin/ferret_paths.sh    (or wherever your system mgr. has put it)
## . $HOME/my_ferret_paths.sh

## These are the environment variables you may wish to customize.
## They are currently set up on the assumption that all your FERRET
## work is done in the directory $HOME/ferret .

   export FER_GO="$FER_GO $HOME/ferret"

   export FER_DATA="$FER_DATA $HOME/ferret"

   export FER_DESCR="$FER_DESCR $HOME/ferret"

   export FER_GRIDS="$FER_GRIDS $HOME/ferret"

   export FER_MODEL_RUNS="$HOME/ferret/model_runs"

