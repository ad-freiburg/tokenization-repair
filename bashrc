if [ -f /etc/bash_completion ]; then source /etc/bash_completion; fi
complete -W "\`grep -oE '^[a-zA-Z0-9_.-]+:([^=]|$)' ?akefile | sed 's/[^a-zA-Z0-9_.-]*$//'\`" make
echo "-------------------------------------------------------"
echo "| Tokenization Repair in the Presence of Misspellings |"
echo "-------------------------------------------------------"
echo
echo "Welcome to the Docker container of the paper 'Tokenization Repair in the Presence of Misspellings'."
echo "Type \"make help\" to list possible targets."
