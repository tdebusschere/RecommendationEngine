library(mailR)
library(optparse)

option_list = list(
  make_option(c("-d", "--directory"), type="character", default="", 
              help="directory", metavar="character"),
  make_option(c("-c", "--commit"), type='character',default="",
              help="commit", metavar="character"),
  make_option(c("-b", "--branch"), type='character',default="staging",
              help="staging", metavar="character"),
  make_option(c("-m", "--message"), type='character', default='',
              help="message")
);
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

body = paste0('problem with commit:', opt$commit," on branch:",opt$branch,", please find the logs included,
              the problem was located at ", opt$message)

directory = paste0(opt$directory)
print(directory)
#setwd(directory)
data = list.files(directory,all.files=TRUE)


attachmentObjects = vector()
attachmentNames = vector()
attachmentDescriptions=vector()
counter = 1
for (x in data){
  if(nchar(x)>2)
  {
    attachmentObjects[counter] = paste0(directory,'//',x)
    attachmentNames[counter] = x
    attachmentDescriptions[counter] = x
    counter = counter + 1
  }
}


sender <- "xinwangds@gmail.com"
recipients <- c("tom_tong@xinwang.com.tw"," jimmy_yin@xinwang.com.tw")
send.mail(from = sender,
          to = recipients,
          cc= sender,
          subject = "Build of staging has failed",
          body = body,
          encoding = "utf-8",
          smtp = list(host.name = "smtp.gmail.com", port = 465, 
                      user.name = "xinwangds@gmail.com",            
                      passwd = '', ssl = TRUE),
          authenticate = TRUE,
          send = TRUE,
          attach.files = attachmentObjects,
          attach.names = attachmentNames,
          attach.descriptions = attachmentDescriptions)
