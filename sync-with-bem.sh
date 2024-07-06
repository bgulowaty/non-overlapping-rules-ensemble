SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
DESTINATION="bogul@ui.wcss.pl:/home/bogul/experiments"

echo "syncing $SCRIPTPATH \nto\n $DESTINATION"

rsync -av --exclude '.ipynb_checkpoints' --exclude '__pycache__' --exclude '.venv' --filter="dir-merge,- .gitignore" $SCRIPTPATH $DESTINATION

