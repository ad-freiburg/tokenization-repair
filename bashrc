if [ -f /etc/bash_completion ]; then source /etc/bash_completion; fi
complete -W "\`grep -oE '^[a-zA-Z0-9_.-]+:([^=]|$)' ?akefile | sed 's/[^a-zA-Z0-9_.-]*$//'\`" make
echo "-----------------------"
echo "| TOKENIZATION REPAIR |"
echo "-----------------------"
echo
echo "Welcome to the Docker from the Tokenization Repair paper."
echo "Type \"make help\" to list possible targets."
