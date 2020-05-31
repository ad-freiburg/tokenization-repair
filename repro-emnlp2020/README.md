# EMNLP 2020 Repro instructions

The URL http://emnlp2020-367.hopto.org currently routes to filicudi.informatik.privat 
From there it is currently proxied to http://tarka.informatik.privat:20367 
The configuration for this is in /etc/apache2/sites-available/repro-emnlp2020 
If you change something in this configuration and want that change to become active: sudo /etc/init.d/apache2 restart
If you don't have sudo rights, pray or open the window and cry for help

To simply serve all files in a certain directory on this port: python3 -m http.simple 20367
If you see a web page under http://emnlp2020-367.hopto.org the server is already running
