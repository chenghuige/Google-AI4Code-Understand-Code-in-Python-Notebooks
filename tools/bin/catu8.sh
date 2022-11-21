perl -ane 'use Encode; print encode("gbk", decode("utf-8",$_));'
