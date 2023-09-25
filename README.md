# general_purpose_image_crawler
This is a general purpose image crawler that works for most websites
This program will crawl website and download images at deisired size in separate processes. 
1. It will save visited URLs and downloaded images into 2 ini files. Make sure you delete them if you want to crawl the whole website again ignoring links visited before.
2. A file called excepts.ini with all unwanted urls can prevent crawling into those links. Each line is an excepted link in format like /football/. A sample ini can be generated with the command --excepts_sample. You may edit the excepts.ini in the process of crawling, the program will reload it every 200 rounds, the same case with censor.ini.
3. A file called censor.ini with all unwanted keywords in the link can prevent crawling into those links. Unlike excepts.ini, censor.ini search the whole link, not only the start.
4. A log file will record the progress of crawling.
5. Normally if your url has certain subdomain of a website, the crawler will lock into this subdomain. Eg. www.some_website.com/football, the crawler will not crawl the whole website, but only the subdomain /football. To cancel this, use --no_lock. Or you may add custom locks with --custom_lock, to add more subdomains to crawl. You may add multiple subdomains. eg. --custom_lock /tennis/ /basketball/ /golf/
6. Please be noted that queue size does not work in Mac OS. It will always show -1.
7. Please be noted that some websites use Javascript to generate pages, and we don't interpret Javascript. So those websites will not be crawled.
8. Many websites may enforce anit-crawling acts, we did not take any measures to prevent this.
9. By default, crawling will work in width first manner, which means it will finish crawling current page first before diving deeper, you may change it by --depth_first.
10. You may watch the crawling process and add some excepts to excepts.ini by echo -e '/some_sub_domain/' | tee -a /path_of_this_program/excepts.ini. It will take effect in next 200 rounds. This is very useful to save time.
11. (4 ) 312/2025, Q: 47 E: 0 I: 815 D: 324, 4 is depth, 312 is current depth urls left in stack, 2025 is total stack to crawl, Q: is length of image urls still in the queue to be downloaded by download workers. E: is connection errors, I: is total image links found, D: is images already downloaded
12. Crawling a large website may take several days, so it's a good practice to run this program in a tmux session. `/usr/bin/tmux new-session -s web -d '/bin/bash'`, and then `tmux attach-session -t web`. Also, you may watch the processes of this program by `ps -eo pid,pcpu,stat,wchan:20,args | grep --color=auto image`.
